# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Implement a multiprocess PPOCritic
"""

import torch
import re
import gc

from verl import DataProto
from verl.workers.comet import BaseCOMETModel


__all__ = ['DataParallelCOMET']


class DataParallelCOMET(BaseCOMETModel):

    def __init__(self, config, comet_module, tokenizer):
        super().__init__(config=config)
        self.comet_model = comet_module 
        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)
        self.tokenizer = tokenizer
        print(self.config)
        self.val_batch_size = self.config.get('val_batch_size')
        if not self.val_batch_size:
            self.val_batch_size = self.config.get("forward_micro_batch_size", 16)
        print(f"dp_comet.py initialized with val_batch_size: {self.val_batch_size}")
     
    def _forward_micro_batch(self, batch, batch_size):
        # response_length = micro_batch['responses'].size(-1)
        # gc.collect()
        # gc.collect(2)
        print(f"dp_comet.py forward micro_batch: {batch_size}")
        comet_output = self.comet_model.predict(batch, batch_size=batch_size, gpus=1, num_workers=0)
        
        # return comet_output.scores #for example: [0.84, 0.77, ...]
        return [float(score) * 100 for score in comet_output.scores]
        

    def extract_translation(self, solution_str: str):
        """
        Extracts the final answer from the model's response string.
        
        Args:
            solution_str: Raw response string from the language model
            
        Returns:
            Tuple containing (extracted_answer, processed_string)
        """
        # --- Remove all <think>...</think> blocks ---
        return re.sub(r"<think>.*?</think>", "", solution_str, flags=re.DOTALL).strip()

    def compute_comet_rm(self, data: DataProto) -> torch.Tensor:
        gc.collect()
        triplet_list = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]


            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            answer_text = self.extract_translation(sequences_str)

            src_text = data_item.non_tensor_batch['extra_info']['src']
            tgt_text = data_item.non_tensor_batch['reward_model']['ground_truth']

            if answer_text:
                new_item = {"src":src_text, "mt":answer_text, "ref":tgt_text}
            else:
                new_item = {"src":src_text, "mt":"None", "ref":tgt_text}
            triplet_list.append(new_item)

        micro_batch_size = data.meta_info['micro_batch_size']
        # print(f"dp_comet.py compute micro_batches: {micro_batches}")
        with torch.no_grad():
            scores = self._forward_micro_batch(triplet_list, batch_size=micro_batch_size)

        reward_tensor = torch.tensor(scores, dtype=torch.float32).unsqueeze(1)

        return reward_tensor

    def compute_valid_comet(self, data: DataProto) -> torch.Tensor:

        triplet_list = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]


            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            answer_text = self.extract_translation(sequences_str)

            src_text = data_item.non_tensor_batch['extra_info']['src']
            tgt_text = data_item.non_tensor_batch['reward_model']['ground_truth']

            new_item = {"src":src_text, "mt":answer_text, "ref":tgt_text}
            triplet_list.append(new_item)


        with torch.no_grad():
            scores = self._forward_micro_batch(triplet_list, batch_size=self.val_batch_size)

        reward_tensor = torch.tensor(scores, dtype=torch.float32).unsqueeze(1)

        return reward_tensor

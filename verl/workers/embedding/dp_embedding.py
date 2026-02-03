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

import torch
import torch.nn.functional as F
from torch import Tensor

from verl import DataProto
from verl.workers.embedding import BaseEmbeddingModel

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]



class DataParallelEmbdding(BaseEmbeddingModel):

    def __init__(self, config, model, tokenizer):
        super().__init__(config=config)
        self.model = model
        self.tokenizer = tokenizer
        print(self.config)
        self.batch_size = self.config.get("forward_micro_batch_size", 16)
        print(f"dp_embedding.py initialized with val_batch_size: {self.batch_size}")
     
    def _forward_micro_batch(self, pre_list, tgt_list):
        print(f"dp_embedding.py forward micro_batch: {self.batch_size}")

        input_texts = pre_list + tgt_list
        batch_dict = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )

        batch_dict.to(self.model.device)
        outputs = self.model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        
        embeddings = F.normalize(embeddings, p=2, dim=1)
        x, y = torch.chunk(embeddings, chunks=2, dim=0)
        return (x * y).sum(dim=1) * 100.0
        

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

    def compute_embed_rm(self, data: DataProto) -> torch.Tensor:
        pre_list = []
        tgt_list = []

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

            tgt_text = data_item.non_tensor_batch['reward_model']['ground_truth']


            pre_list.append(answer_text)
            tgt_list.append(tgt_text)

        scores = ()
        with torch.no_grad():
            for i in range(0, len(pre_list), self.batch_size):
                st = i
                ed = min(i + self.batch_size, len(pre_list))
                scores += (self._forward_micro_batch(pre_list[st:ed], tgt_list[st:ed]),)

        reward_tensor = torch.cat(scores, dim=0).unsqueeze(1)

        return reward_tensor

    def compute_valid_embed(self, data: DataProto) -> torch.Tensor:
        pre_list = []
        tgt_list = []

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

            tgt_text = data_item.non_tensor_batch['reward_model']['ground_truth']

            pre_list.append(answer_text)
            tgt_list.append(tgt_text)


        scores = ()
        with torch.no_grad():
            for i in range(0, len(pre_list), self.batch_size):
                st = i
                ed = min(i + self.batch_size, len(pre_list))
                scores += (self._forward_micro_batch(pre_list[st:ed], tgt_list[st:ed]),)

        reward_tensor = torch.cat(scores, dim=0).unsqueeze(1)

        return reward_tensor

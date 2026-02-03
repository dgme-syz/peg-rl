# Copyright 2024 Bytedance Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict, Counter
from typing import Any
import copy

import torch

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
from verl.utils.reward_score.mt_score import compute_score_corpus, compute_score_val_bleu


def make_single_batch(batch):
    """
    Ensure TensorDict has batch_size == (1,)
    """
    if hasattr(batch, "batch_size"):
        if len(batch.batch_size) == 0:
            batch = batch.unsqueeze(0)
        elif batch.batch_size[0] != 1:
            raise ValueError(f"Unexpected batch_size: {batch.batch_size}")
    else:
        raise TypeError("batch must be a TensorDict")
    return batch

def make_single_non_tensor_batch(non_tensor_batch):
    """
    Ensure each value in non_tensor_batch is a list.
    If value is not a list, wrap it into a single-element list.
    """
    if not isinstance(non_tensor_batch, dict):
        raise TypeError(f"Expected dict, got {type(non_tensor_batch)}")

    out = {}
    for k, v in non_tensor_batch.items():
        if isinstance(v, list):
            assert len(v) == 1, f"Key {k} expects single element, got {len(v)}"
            out[k] = v
        else:
            out[k] = [copy.deepcopy(v)]
    return out

def normalize_single_dataproto(item: DataProto) -> DataProto:
    item.batch = make_single_batch(item.batch)
    item.non_tensor_batch = make_single_non_tensor_batch(item.non_tensor_batch)
    return item



@register("mt_train")
class MtTrainRewardManager(AbstractRewardManager):
    """Reward manager for machine translation (training stage)."""

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score=None,
        reward_fn_key: str = "data_source",
    ) -> None:
        """
        Initialize the MtTrainRewardManager.

        Args:
            tokenizer: Tokenizer used to decode token IDs into text.
            num_examine: Number of batches of decoded responses to print for debugging.
            compute_score: Function to compute reward scores.
            reward_fn_key: Key to access data source in non-tensor batch data.
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.print_num = 20

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """Compute MT rewards for each sample in the batch."""
        if "rm_scores" in data.batch:
            raise NotImplementedError(
                "MT reward model score combination not supported; use BLEU or COMET scores instead."
            )

        count = Counter()
        reward_tensor = torch.zeros(
            data.batch["responses"].shape,
            dtype=torch.float32,
            device=data.batch["responses"].device,
        )
        reward_extra_info = defaultdict(list)
        printed_data_sources: dict[str, int] = {}


        # exp: if right, acl, else failure
        tree_parent = {}
        scores_list = []

        for i, data_item in enumerate(data):
            # --- Decode tokens ---

            prompt_ids = data_item.batch["prompts"]
            response_ids = data_item.batch["responses"]
            attn_mask = data_item.batch["attention_mask"]

            prompt_len = prompt_ids.shape[-1]
            valid_prompt_len = int(attn_mask[:prompt_len].sum())
            valid_response_len = int(attn_mask[prompt_len:].sum())

            prompt_str = self.tokenizer.decode(
                prompt_ids[-valid_prompt_len:], skip_special_tokens=True
            )
            response_str = self.tokenizer.decode(
                response_ids[:valid_response_len], skip_special_tokens=True
            )

            non_tensor = data_item.non_tensor_batch
            depth = non_tensor.get("depth", 1)

            if non_tensor.get("depth", None) == 1:
                uid = non_tensor.get("own_uid")
                tree_parent[uid] = i

            ground_truth = non_tensor["reward_model"]["ground_truth"]
            data_source = non_tensor[self.reward_fn_key]
            translation_raw = non_tensor["last_response"] if non_tensor.get("last_response", None) else None
            lg_pair = f"{non_tensor['extra_info']['src_lang']}-{non_tensor['extra_info']['tgt_lang']}"

            metric_scores = [
                float(v) for k, v in data_item.batch.items() if k.endswith("_score")
            ]

            count[depth] += 1
            # --- Compute reward ---
            score = self.compute_score(
                metric_scores=metric_scores,
                lang_pair=lg_pair,
                prompt_str=prompt_str,
                solution_str=response_str,
                ground_truth=ground_truth,
                translation_raw=translation_raw,
                print_ok=count[depth] <= self.print_num,
                resp_token_length=valid_response_len
            )


            scores_list.append(score)
            printed_data_sources.setdefault(data_source, 0)
            if printed_data_sources[data_source] < self.num_examine:
                printed_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

            if isinstance(score, dict):
                reward = score.get("score", 0.0)
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_len - 1] = reward

        exp_mode = False
        if not exp_mode:
            return (
                {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
                if return_dict
                else reward_tensor
            )
    
        great_dict, bad_dict = defaultdict(DataProto), defaultdict(DataProto)
        great_score, bad_score = {}, {}
        if exp_mode:
            for i, data_item in enumerate(data):
                if data_item.non_tensor_batch.get("depth", None) != 2:
                    continue

                prompt_ids = data_item.batch["prompts"]
                response_ids = data_item.batch["responses"]
                attn_mask = data_item.batch["attention_mask"]

                prompt_len = prompt_ids.shape[-1]

                uid = data_item.non_tensor_batch.get("uid")
                par = data[tree_parent[uid]]


                par_prompt_len = par.batch["prompts"].shape[-1]
                sample_uid = par.non_tensor_batch.get("uid")

                if (sample_uid not in great_score or scores_list[i] > great_score[sample_uid]) and scores_list[i] > 0:
                    great_score[sample_uid] = scores_list[i]
                    item = copy.deepcopy(data_item)
                    item.batch = data_item.batch.detach().clone()
                    item.non_tensor_batch = copy.deepcopy(data_item.non_tensor_batch)


                    item.non_tensor_batch["uid"] = sample_uid
                    item.non_tensor_batch["own_uid"] = "-1"
                    item.non_tensor_batch["depth"] = 1
                    item.batch["prompts"] = par.batch["prompts"].detach().clone()
                    item.batch["input_ids"] = torch.cat((item.batch["prompts"].detach().clone(), item.batch["responses"].detach().clone()), dim=0)
                    item.batch["attention_mask"] = torch.cat(
                        [par.batch["attention_mask"][:par_prompt_len].detach().clone(), item.batch["attention_mask"][prompt_len:].detach().clone()],
                        dim=0
                    )
                    item.batch["position_ids"] = torch.cat(
                        [par.batch["position_ids"][:par_prompt_len].detach().clone(), item.batch["position_ids"][prompt_len:].detach().clone()],
                        dim=0
                    )
                    great_dict[sample_uid] = item

                if (sample_uid not in bad_score or scores_list[i] < bad_score[sample_uid]) and scores_list[i] > 0:
                    bad_score[sample_uid] = scores_list[i]
                    item = copy.deepcopy(data_item)
                    item.batch = data_item.batch.detach().clone()
                    item.non_tensor_batch = copy.deepcopy(data_item.non_tensor_batch)

                    item.non_tensor_batch["uid"] = sample_uid
                    item.non_tensor_batch["own_uid"] = "-1"
                    item.non_tensor_batch["depth"] = 1

                    item.batch["prompts"] = par.batch["prompts"].detach().clone()
                    item.batch["input_ids"] = torch.cat((item.batch["prompts"].detach().clone(), item.batch["responses"].detach().clone()), dim=0)
                    item.batch["attention_mask"] = torch.cat(
                        [par.batch["attention_mask"][:par_prompt_len].detach().clone(), item.batch["attention_mask"][prompt_len:].detach().clone()],
                        dim=0
                    )
                    item.batch["position_ids"] = torch.cat(
                        [par.batch["position_ids"][:par_prompt_len].detach().clone(), item.batch["position_ids"][prompt_len:].detach().clone()],
                        dim=0
                    )
                    bad_dict[sample_uid] = item

        def check(data_item: DataProto, score):
            prompt_ids = data_item.batch["prompts"]
            response_ids = data_item.batch["responses"]
            input_ids = data_item.batch["input_ids"]
            attn_mask = data_item.batch["attention_mask"]
            position_ids = data_item.batch["position_ids"]

            prompt_len = prompt_ids.shape[-1]
            valid_prompt_len = int(attn_mask[:prompt_len].sum())
            valid_response_len = int(attn_mask[prompt_len:].sum())

            input_str = self.tokenizer.decode(
                input_ids, skip_special_tokens=True
            )

            prompt_str = self.tokenizer.decode(
                prompt_ids[-valid_prompt_len:], skip_special_tokens=True
            )
            response_str = self.tokenizer.decode(
                response_ids[:valid_response_len], skip_special_tokens=True
            )

            valid_pos_ids = position_ids[attn_mask.bool()]
            assert len(valid_pos_ids) == valid_prompt_len + valid_response_len
            print(
                f"======================================= INFO ============================\n\n"
                f"Preview one added mt data: \n"
                f"{data_item}\n\n"
                f"prompt: {prompt_str}\n"
                f"response: {response_str}\n"
                f"input: {input_str}\n"
                f"score: {score}\n"
                f"======================================= INFO ============================\n\n"
            )



        def get_valid_response_length(data_item: DataProto):
            prompt_ids = data_item.batch["prompts"]
            prompt_len = prompt_ids.shape[-1]
            attn_mask = data_item.batch["attention_mask"]
            valid_response_len = int(attn_mask[prompt_len:].sum())

            return valid_response_len

        extra_mt_dataproto = [data]

        while (len(great_dict) + len(bad_dict)) % 8:
            if len(bad_dict) > 0:
                bad_dict.popitem()
                continue
            if len(great_dict) > 0:
                great_dict.popitem()
                continue
        
        if (len(great_dict) + len(bad_dict)) > 0:
            dim = data.batch["responses"].shape[-1]
            new_reward_tensor = torch.zeros(
                (len(great_dict) + len(bad_dict), dim),
                dtype=torch.float32,
                device=data.batch["responses"].device,
            )
            
            for i, sample_uid in enumerate(great_dict.keys()):
                great_item = great_dict[sample_uid]
                bad_item = bad_dict[sample_uid]

                if i == 0: 
                    check(great_item, great_score[sample_uid])
                    check(bad_item, bad_score[sample_uid])

                extra_mt_dataproto.extend(
                    [normalize_single_dataproto(great_item), normalize_single_dataproto(bad_item)]
                )

                new_reward_tensor[2 * i, get_valid_response_length(great_item) - 1] = great_score[sample_uid]
                new_reward_tensor[2 * i + 1, get_valid_response_length(bad_item) - 1] = bad_score[sample_uid]

            # combine data and reward_score
            data = DataProto.concat(extra_mt_dataproto) # data need return back
            print(data.batch)
            reward_tensor = torch.cat(
                (reward_tensor, new_reward_tensor)
            )

        return reward_tensor, data


@register("mt_val")
class MtValRewardManager(AbstractRewardManager):
    """Reward manager for machine translation (validation stage)."""

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score=None,
        reward_fn_key: str = "data_source",
    ) -> None:
        """
        Initialize the MtValRewardManager.

        During validation, BLEU or similar metric is computed directly.

        Args:
            tokenizer: Tokenizer used to decode token IDs into text.
            num_examine: Number of batches of decoded responses to print for debugging.
            compute_score: Function to compute validation score (e.g., BLEU).
            reward_fn_key: Key to access the data source in the non-tensor batch.
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        # we do not need ant params 
        # if rewardmanager can use settings from config, it is easy
        # now we use custom func args to control score caclulation's params in training
        self.compute_score = compute_score_val_bleu 
        self.compute_corpus_bleu = compute_score_corpus
        self.reward_fn_key = reward_fn_key
        self.print_num = 20

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """Compute validation BLEU or other metric-based reward."""
        if "rm_scores" in data.batch:
            raise NotImplementedError(
                "MT validation with reward model score not supported; use BLEU or COMET."
            )

        reward_tensor = torch.zeros(
            data.batch["responses"].shape,
            dtype=torch.float32,
            device=data.batch["responses"].device,
        )
        reward_extra_info = defaultdict(list)
        printed_data_sources: dict[str, int] = {}

        sol = defaultdict(list)
        ref = defaultdict(list)
        idx = defaultdict(list)

        global_lang_pair = None

        count = Counter()
        for i, data_item in enumerate(data):
            prompt_ids = data_item.batch["prompts"]
            response_ids = data_item.batch["responses"]
            attn_mask = data_item.batch["attention_mask"]

            prompt_len = prompt_ids.shape[-1]
            valid_prompt_len = int(attn_mask[:prompt_len].sum())
            valid_response_len = int(attn_mask[prompt_len:].sum())

            prompt_str = self.tokenizer.decode(
                prompt_ids[-valid_prompt_len:], skip_special_tokens=True
            )
            response_str = self.tokenizer.decode(
                response_ids[:valid_response_len], skip_special_tokens=True
            )

            non_tensor = data_item.non_tensor_batch
            depth = non_tensor.get("depth", 1)
            ground_truth = non_tensor["reward_model"]["ground_truth"]
            data_source = non_tensor[self.reward_fn_key]
            lg_pair = f"{non_tensor['extra_info']['src_lang']}-{non_tensor['extra_info']['tgt_lang']}"
            if global_lang_pair is None:
                global_lang_pair = lg_pair

            if valid_response_len == len(attn_mask[prompt_len:]):
                ## token budget is full
                response_str = "null"
            sol[data_source].append(response_str)
            ref[data_source].append(ground_truth)
            idx[data_source].append(i)

            count[depth] += 1

            score = self.compute_score(
                solution_str=response_str,
                ground_truth=ground_truth,
                lang_pair=lg_pair,
                print_ok=count[depth] <= self.print_num,
            )

            if not isinstance(score, dict):
                score = {"score": score}

            # Attach any validation metrics
            for key, value in data_item.batch.items():
                if key.endswith("_valid"):
                    score[key] = float(value)

            reward = score.get("score")
            for key, value in score.items():
                reward_extra_info[key].append(value)

            reward_tensor[i, valid_response_len - 1] = reward

            printed_data_sources.setdefault(data_source, 0)
            if printed_data_sources[data_source] < self.num_examine:
                printed_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                for key, value in score.items():
                    print(f"[{key}]", value)

        # Compute corpus-level BLEU if applicable
        if self.compute_corpus_bleu is not None and global_lang_pair is not None:
            for data_source in sol.keys():
                sol_list = sol[data_source]
                ref_list = ref[data_source]
                corpus_score, mode = self.compute_corpus_bleu(
                    solution_str=sol_list,
                    ground_truth=ref_list,
                    lang_pair=global_lang_pair,
                    print_ok=True
                )
                if len(reward_extra_info[f"corpus_{mode}"]) == 0:
                    reward_extra_info[f"corpus_{mode}"] = [0 for _ in range(len(data))]

                for i in idx[data_source]:
                    reward_extra_info[f"corpus_{mode}"][i] = corpus_score

                print(f"[corpus_{mode}_score - {data_source}]", corpus_score)

        return (
            {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info} if return_dict else reward_tensor
        )
    

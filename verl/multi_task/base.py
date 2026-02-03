from abc import ABC
from typing import Any, Dict, List, Tuple
import re
import uuid
import copy
import textwrap
import random

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.fs import copy_to_local
from verl.utils import hf_tokenizer
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
 



LANG_DICT = {
    "en": "English", "zh": "Chinese (Simplified)", "hu": "Hungarian", "es": "Spanish", "fr": "French", "de": "German", "ru": "Russian", "ja": "Japanese", "th": "Thai", "sw": "Swahili", "bn": "Bengali", "te": "Telugu", "ar": "Arabic", "ko": "Korean", "vi": "Vietnamese", "cs": "Czech", "sr": "Cyrillic Serbian"
}

LANG_DICT.update({
    "ha": "Hausa",
    "om": "Oromo",
    "so": "Somali",
    "am": "Amharic",
    "he": "Hebrew",
    "mt": "Maltese",
    "km": "Khmer",
    "jv": "Javanese",
    "id": "Indonesian",
    "ms": "Malay",
    "mi": "Maori",
    "ceb": "Cebuano",
    "tl": "Tagalog",
    "kn": "Kannada",
    "ml": "Malayalam",
    "ta": "Tamil",
    "hy": "Armenian",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bs": "Bosnian",
    "hr": "Croatian",
    "mk": "Macedonian",
    "pl": "Polish",
    "sk": "Slovak",
    "sl": "Slovenian",
    "uk": "Ukrainian",
    "cy": "Welsh",
    "ga": "Irish",
    "is": "Icelandic",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "af": "Afrikaans",
    "lb": "Luxembourgish",
    "nl": "Dutch",
    "el": "Greek",
    "as": "Assamese",
    "gu": "Gujarati",
    "hi": "Hindi",
    "mr": "Marathi",
    "ne": "Nepali",
    "or": "Odia",
    "pa": "Punjabi",
    "sd": "Sindhi",
    "ur": "Urdu",
    "fa": "Persian",
    "ku": "Kurdish",
    "ps": "Pashto",
    "tg": "Tajik",
    "ast": "Asturian",
    "ca": "Catalan",
    "gl": "Galician",
    "it": "Italian",
    "oc": "Occitan",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ka": "Georgian",
    "lo": "Lao",
    "mn": "Mongolian",
    "wo": "Wolof",
    "ln": "Lingala",
    "ns": "Northern Sotho",
    "lg": "Luganda",
    "ny": "Nyanja",
    "sn": "Shona",
    "umb": "Umbundu",
    "xh": "Xhosa",
    "yo": "Yoruba",
    "zu": "Zulu",
    "ig": "Igbo",
    "kam": "Kamba",
    "ff": "Fulani",
    "luo": "Dholuo",
    "kea": "Kabuverdianu",
    "zhtrad": "Traditional Chinese",
    "my": "Burmese",
    "uz": "Uzbek",
    "kk": "Kazakh",
    "ky": "Kyrgyz",
    "az": "Azerbaijani",
    "tr": "Turkish",
    "et": "Estonian",
    "fi": "Finnish",
})

class MultiTaskWorkflow(ABC):
    """Abstract base class for defining multi-task workflows."""

    actor_rollout_wg: Any
    config: Any
    gen_batch_output: DataProto

    def __init__(self, config: Any):
        # gen_batch_output: generated batch from language model, bsz*n
        self.config = config

    def set(
        self,
        gen_batch_output: DataProto,
        actor_rollout_wg: Any,
    ):
        self.gen_batch_output = gen_batch_output
        self.actor_rollout_wg = actor_rollout_wg


    def work(self, **kwargs) -> DataProto:
        raise NotImplementedError


class MTWorkflow(MultiTaskWorkflow):
    """
    Implements a multi-turn machine translation workflow:
    1. Initial translation (MT).
    2. Post-editing/Refinement of the translation.
    """

    def __init__(self, config: Any):
        super().__init__(config)
        tokenizer_path = copy_to_local(self.config.workflow.get("tokenizer_path"))
        self.tokenizer: PreTrainedTokenizerBase = hf_tokenizer(tokenizer_path)
        self.repeat_times = self.config.workflow.get("repeat_times")
        self.max_prompt_length = self.config.data.get("max_prompt_length", 1024)
        self.truncation = self.config.data.get("truncation", "error")
        self.use_test_prompt = self.config.workflow.get("use_test_prompt", False)
        self.mt_only = self.config.workflow.get("mt_only", False)
        self.test_mt_only = self.config.workflow.get("test_mt_only", False)
        self.local_steps = 0
        self.data_divisor = self.config.workflow.get("data_divisor", 1)
        self.dynamic_mode = self.config.workflow.get("dynamic_mode", False)

    @staticmethod
    def _extract_translation(solution_str: str) -> str | None:
        """
        Extracts the final answer from the model's response string by removing <think> blocks.
        """
        # --- Remove all <think>...</think> blocks ---
        processed_str = re.sub(r"<think>.*?</think>", "", solution_str, flags=re.DOTALL).strip()
        return processed_str if "<think>" not in processed_str else None

    def _build_recheck_prompt(self, target_lang: str, src_text: str, pred_text: str) -> List[Dict[str, str]]:
        """Builds the prompt for the recheck/refinement step."""
        prompts = {
            "zh": textwrap.dedent(
                f"""
                给定源文：'{src_text}'，和它的翻译草稿：'{pred_text}'
                必须先理解源文，然后参考以下标准对草稿进行进一步修改润色

                1. 草稿翻译可能漏翻，请不要遗漏原文的含义
                2. 保证翻译文段读起来流畅，通顺，符合人类表达，可以调整句子的顺序
                3. 请注意仔细理解语境，选择书面语还是口语化表达
                4. 请再检查每个词翻译的意思，是否符合整个语境和现实社会
                5. 请再检查每个句翻译的意思，是否符合整个语境和现实社会
                6. 注意你的润色对象是翻译后的草稿，不是源文
                7. 当你觉得语句读起来困惑的时候，尝试从源文本重新思考
                8. 如果翻译草稿的语言并非中文，请确保你的润色文本为中文
                9. 注意检查，不要遗漏源文的含义，也不要添加补充，也不要尝试在翻译中使用过分的比喻
                10. 可以有思考过程，不要无限思考下去，最终回复中仅输出你润色后的内容

                请返回你最后的润色翻译文本，不要输出多余内容。
                """
            ).strip(),
            "en": textwrap.dedent(
                f"""
                Given the source text: '{src_text}' and its draft translation: '{pred_text}', please refine and polish the draft translation according to the following guidelines:

                1. The draft translation may have omissions — do not leave out any meaning from the source text.
                2. Ensure the translation reads smoothly and naturally, consistent with human expression; you may adjust sentence order if needed.
                3. Carefully understand the context, and decide whether to use formal or colloquial language.
                4. Recheck the meaning of each word to ensure it fits the overall context and real-world usage.
                5. Recheck the meaning of each sentence to ensure it fits the overall context and real-world usage.
                6. Note that your task is to polish the translated draft, not the source text.
                7. When a sentence feels confusing, reconsider it from the perspective of the source text.
                8. If the draft translation is not in English, make sure your refined version is in English.
                9. Do not include any extra reasoning or commentary — only output your polished translation.

                Please return only the final refined translation text, without any additional content.
                """
            ).strip(),
            "fi": textwrap.dedent(
                f"""
                Kun sinulle annetaan lähdeteksti: '{src_text}' ja sen käännösluonnos: '{pred_text}',
                sinun tulee ensin ymmärtää lähdeteksti ja sen jälkeen muokata ja viimeistellä luonnosta
                seuraavien periaatteiden mukaisesti.

                1. Käännösluonnoksessa saattaa olla puutteita; älä jätä pois mitään alkutekstin merkitystä.
                2. Varmista, että käännetty teksti on sujuvaa, luonnollista ja ihmisen tapaista; voit tarvittaessa
                   muuttaa virkkeen sanajärjestystä.
                3. Kiinnitä erityistä huomiota kontekstiin ja valitse tilanteeseen sopiva tyyli, olipa se sitten
                   kirjakielinen tai puhekielisempi.
                4. Tarkista jokaisen sanan merkitys ja varmista, että se sopii sekä lauseyhteyteen että todelliseen maailmaan.
                5. Tarkista jokaisen virkkeen merkitys ja varmista, että se vastaa koko kontekstia ja todellisuutta.
                6. Muista, että viimeisteltävä kohde on käännösluonnos, ei alkuperäinen lähdeteksti.
                7. Jos jokin kohta tuntuu epäselvältä, palaa lähdetekstiin ja mieti sitä uudelleen.
                8. Jos käännösluonnos ei ole kiinaksi, varmista että viimeistelty tekstisi on kiinankielinen.
                9. Huolehdi siitä, ettet jätä pois alkutekstin merkityksiä, älä lisää ylimääräistä sisältöä
                   äläkä käytä ylenpalttisia vertauksia käännöksessä.
                10. Voit näyttää ajatteluprosessin, mutta älä jatka sitä loputtomiin; lopullisessa vastauksessa
                    tulee olla vain viimeistelemäsi teksti.

                Palauta lopuksi vain viimeistelty käännöksesi, älä mitään ylimääräistä.
                """
            ).strip(),
            "test": textwrap.dedent(
                f"""
                Given the source text: 
                
                {src_text}
                
                Improve the following draft {LANG_DICT.get(target_lang, target_lang)} translation into a high-quality {LANG_DICT.get(target_lang, target_lang)} version, without explanations:

                {pred_text}
                """
            ).strip(),
        }
        
        lang_key = next((key for key in prompts if key in target_lang), None)
        if lang_key is None:
            if not self.use_test_prompt:
                raise ValueError(f"Unsupported target language: {target_lang}. Supported: {list(prompts.keys())}")

        if self.use_test_prompt:
            lang_key = "test"

        return [{"role": "user", "content": prompts[lang_key]}]

    def _prepare_recheck_inputs(
        self, gen_batch_output: DataProto, repeat_times: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str], List[int]]:
        """
        Process the initial generation output to prepare inputs for the recheck step.
        This involves decoding, extracting answers, and re-tokenizing for the next turn.
        """
        input_ids_list, attention_mask_list, position_ids_list = [], [], []
        last_responses, valid_indices = [], []

        uid_list = set()
        for data_item in gen_batch_output:
            uid = data_item.non_tensor_batch["uid"]
            if not isinstance(data_item.non_tensor_batch.get("last_response", None), str):
                uid_list.add(uid)
        print(f"Preparing recheck inputs for {len(uid_list)}/{len(gen_batch_output)} unique samples.")
        # select len(uid_list) // repeat_times valid samples
        selected_uids = set(random.sample(list(uid_list), max(1, len(uid_list) // repeat_times)))

        for i, data_item in enumerate(gen_batch_output):
            uid = data_item.non_tensor_batch["uid"]
            if uid not in selected_uids:
                continue
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            if isinstance(data_item.non_tensor_batch.get("last_response", None), str):
                # fixed data
                continue

            sequences_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            answer_str = self._extract_translation(sequences_str)

            valid_indices.append(i)
            if not answer_str:
                answer_str = "null"
                print(f"Warning: sample {i: }, len={valid_response_length} {sequences_str} \n has no extracted translation, set to default 'null'")
            src_text = data_item.non_tensor_batch["extra_info"]["src"]
            tgt_lang = data_item.non_tensor_batch["extra_info"]["tgt_lang"]

            def recheck_tokenized(answer_str):

                recheck_chat = self._build_recheck_prompt(
                    target_lang=tgt_lang, src_text=src_text, pred_text=answer_str
                )
                recheck_text = self.tokenizer.apply_chat_template(recheck_chat, add_generation_prompt=True, tokenize=False, enable_thinking=self.config.data.apply_chat_template_kwargs.get("enable_thinking", True))
                recheck_prompt_tokenized = self.tokenizer(recheck_text, return_tensors="pt", add_special_tokens=False)
                return recheck_prompt_tokenized, recheck_text

            recheck_prompt_tokenized, recheck_text = recheck_tokenized(answer_str)
            if i == 0:
                print(f"Preview:\nanswer_str:\n{answer_str}\nsource text:\n{src_text}\nReCheck prompt example:\n{recheck_text}")
            if recheck_prompt_tokenized['input_ids'].shape[-1] > self.max_prompt_length:
                print(f"Warning: sample {i: } recheck prompt length {recheck_prompt_tokenized['input_ids'].shape[-1]} exceeds max_prompt_length {self.max_prompt_length}")
                recheck_prompt_tokenized, _ = recheck_tokenized("null")
                answer_str = "null"
            
            last_responses.append(answer_str)
            x, y = verl_F.postprocess_data(
                input_ids=recheck_prompt_tokenized['input_ids'],
                attention_mask=recheck_prompt_tokenized['attention_mask'],
                max_length=self.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.truncation,
            )
            z = compute_position_id_with_mask(y)
            input_ids_list.append(x[0])
            attention_mask_list.append(y[0])
            position_ids_list.append(z[0])

        return (
            torch.stack(input_ids_list),
            torch.stack(attention_mask_list),
            torch.stack(position_ids_list),
            last_responses,
            valid_indices,
        )

    def _align_batch_for_dp(self, indices: List[int], repeat_times: int) -> List[int]:
        """Aligns the batch of indices to be divisible by the data-parallel world size."""
        dp_world_size = self.actor_rollout_wg.world_size
        if dp_world_size <= 0:
            return indices

        num_samples = len(indices) * repeat_times
        remainder = num_samples % dp_world_size
        if remainder == 0:
            return indices

        samples_to_drop = remainder // repeat_times + (1 if remainder % repeat_times != 0 else 0)
        print(f"Aligning batch for DP: dropping {samples_to_drop} samples to make batch size divisible by {dp_world_size}")
        return indices[:-samples_to_drop]

    def _build_post_edit_batch(
        self,
        original_batch: DataProto,
        recheck_input_ids: torch.Tensor,
        recheck_attn_mask: torch.Tensor,
        recheck_pos_ids: torch.Tensor,
        last_responses: List[str],
        uuid,
    ) -> DataProto:
        """Constructs the DataProto object for the post-editing generation step."""
        # Use shallow copy for efficiency, as we are only modifying top-level dicts
        post_edit_batch = copy.deepcopy(original_batch)
        post_edit_batch.batch = original_batch.batch.detach().clone()
        post_edit_batch.non_tensor_batch = copy.deepcopy(original_batch.non_tensor_batch)

        post_edit_batch.batch.pop("prompts")
        post_edit_batch.batch.pop("responses")
        post_edit_batch.batch.update({
            "input_ids": recheck_input_ids,
            "attention_mask": recheck_attn_mask,
            "position_ids": recheck_pos_ids,
        })

        batch_size = len(recheck_input_ids)
        post_edit_batch.non_tensor_batch.update({
            "last_response": np.array(last_responses, dtype=object),
            "uid": uuid,
            "depth": np.full(batch_size, 2, dtype=np.int32)
        })
        return post_edit_batch

    def work(self, repeat_times: int | None = None, concat: bool = True, test: bool = False, pad_size: int = 0) -> DataProto:
        """Executes the full multi-turn workflow."""
        if not test:
            self.local_steps += 1
            if self.dynamic_mode and self.local_steps % 51 == 0 and self.data_divisor // 2 >= 1:
                self.data_divisor = self.data_divisor // 2
                print(f"Adjusting data_divisor to {self.data_divisor} at local step {self.local_steps}")

        repeat_times = repeat_times if repeat_times is not None else self.repeat_times
        # 1. Initial Generation (MT)
        gen_batch_output = self.actor_rollout_wg.generate_sequences(self.gen_batch_output)
        if not test:
            if self.mt_only:
                return gen_batch_output
        else:
            if self.test_mt_only:
                return gen_batch_output
            print("Use post edit evaluation mode.")

        gen_batch_output.non_tensor_batch["own_uid"] = np.array([str(uuid.uuid4()) for _ in range(len(gen_batch_output.batch))], dtype=object)
        # 2. Prepare for Post-Editing Step
        (
            recheck_input_ids,
            recheck_attn_mask,
            recheck_pos_ids,
            last_responses,
            valid_indices,
        ) = self._prepare_recheck_inputs(gen_batch_output[:-pad_size] if pad_size > 0 else gen_batch_output, repeat_times=repeat_times if test else self.data_divisor)

        if not test:
            valid_indices = self._align_batch_for_dp(valid_indices, repeat_times)
        else:
            valid_indices = list(range(len(recheck_input_ids))) 
        # Filter all inputs based on valid indices
        recheck_input_ids = recheck_input_ids[:len(valid_indices)]
        recheck_attn_mask = recheck_attn_mask[:len(valid_indices)]
        recheck_pos_ids = recheck_pos_ids[:len(valid_indices)]
        filtered_last_responses = last_responses[:len(valid_indices)]
        
        # Build the batch for the post-editing step
        post_edit_batch = self._build_post_edit_batch(
            original_batch=gen_batch_output[valid_indices],
            recheck_input_ids=recheck_input_ids,
            recheck_attn_mask=recheck_attn_mask,
            recheck_pos_ids=recheck_pos_ids,
            last_responses=filtered_last_responses,
            uuid=gen_batch_output.non_tensor_batch["own_uid"][valid_indices]
        )

        # Set metadata for the original generation batch
        # update mt_source_name
        gen_batch_output.non_tensor_batch["data_source"] = np.array(
            [f"{x}_mt" for x in gen_batch_output.non_tensor_batch["data_source"]], dtype=object
        )
        depth = []
        last_responses = []
        for x in gen_batch_output:
            if x.non_tensor_batch.get("last_response", None) is not None:
                depth.append(2)
                last_responses.append(x.non_tensor_batch["last_response"])
            else:
                depth.append(1)
                last_responses.append(None)
        gen_batch_output.non_tensor_batch["depth"] = np.array(depth, dtype=np.int32)
        gen_batch_output.non_tensor_batch["last_response"] = np.array(last_responses, dtype=object)

        print(f"Post-edit batch size (before repeat): {len(post_edit_batch.batch)}")
        print(f"Preview of post-editing:\n{gen_batch_output[:2]}")
        
        # 3. Post-Editing Generation
        post_edit_batch = post_edit_batch.repeat(repeat_times=repeat_times, interleave=True)
        size_divisor = self.actor_rollout_wg.world_size
        if test:
            print("Two Stage Pad: before, len post edit batch:", len(post_edit_batch.batch))
            post_edit_batch, pad_size = pad_dataproto_to_divisor(post_edit_batch, size_divisor)
            print("Two Stage Pad: after, len post edit batch:", len(post_edit_batch.batch))
            post_edit_output = self.actor_rollout_wg.generate_sequences(post_edit_batch)
            print("Two Stage Pad: finish, len post edit output:", len(post_edit_output.batch))
            post_edit_output = unpad_dataproto(post_edit_output, pad_size=pad_size)
        else:
            post_edit_output = self.actor_rollout_wg.generate_sequences(post_edit_batch)
        post_edit_output.non_tensor_batch["own_uid"] = np.full(len(post_edit_output.batch), "-1", dtype=object)

        # 4. Combine results
        gen_batch_output.meta_info.pop("timing", None)
        final_output = DataProto.concat([post_edit_output, gen_batch_output])

        print(f"Final combined batch size: {len(final_output.batch)}")
        return final_output




import re
import textwrap
import sacrebleu
import torch
import pyarrow.parquet as pq
from vllm import LLM, SamplingParams
from comet import download_model, load_from_checkpoint
from transformers import AutoTokenizer
from typing import List, Dict
import ray


# =========================
# Prompt builders
# =========================

TP_SIZE = 8

LANG_NAME = {
    "en": "English", "fi": "Finnish", "zh": "Chinese",
    "cs": "Czech", "de": "German", "fr": "French",
    "et": "Estonian", "kk": "Kazakh", "tr": "Turkish",
}

def qwen_chat_input(src_lang_name, tgt_lang_name, src_text):
    user_input = (
        f"Translate the following text into {tgt_lang_name} "
        f"without additional explanations:\n\n"
        f"{src_text}\n\n"
    )
    return [{"role": "user", "content": user_input}]


def build_postedit_prompt(src_text, draft_translation, target_lang_name):
    return textwrap.dedent(
        f"""
        Given the source text:

        {src_text}

        Improve the following draft {target_lang_name} translation
        into a high-quality {target_lang_name} version,
        without explanations:

        {draft_translation}
        """
    ).strip()


# =========================
# Output preprocess
# =========================

def preprocess(text: str) -> str:
    parts = re.split(r'</think\s*>', text, flags=re.IGNORECASE)
    text_after_think = parts[-1] if len(parts) > 1 else text
    match = re.search(
        r'<text\s*>(.*?)</text\s*>',
        text_after_think,
        flags=re.IGNORECASE | re.DOTALL
    )
    if match:
        return match.group(1).strip()
    return text_after_think.strip()


# =========================
# Metrics
# =========================

class chrFpp:
    def __call__(self, responses, references):
        return sacrebleu.corpus_chrf(
            responses,
            [references],
            word_order=2,
            beta=2
        ).score / 100.0


def load_comet_kiwi():
    model_path = download_model("Unbabel/wmt23-cometkiwi-da-xl")
    return load_from_checkpoint(model_path)


@ray.remote(num_gpus=1)
class CometActor:
    def __init__(self):
        model_path = download_model("Unbabel/wmt23-cometkiwi-da-xl")
        self.model = load_from_checkpoint(model_path)
        self.model.eval()

    def predict(self, inputs, batch_size=144):
        scores = []
        with torch.no_grad():
            out = self.model.predict(
                inputs,
                batch_size=batch_size,
                gpus=1
            )
            scores.extend(out["scores"])
        return scores


@ray.remote(num_gpus=TP_SIZE)
class LLMActor:
    def __init__(self, model_name: str, tp_size: int):
        self.llm = LLM(
            model=model_name,
            tokenizer=model_name,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            tensor_parallel_size=tp_size,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def chat(self, prompts, n, temperature=1.0):
        prompt_strs = [
            build_prompt(self.tokenizer, p)
            for p in prompts
        ]
        sampling_params = SamplingParams(
            temperature=temperature,
            n=n,
            max_tokens=512
        )
        outputs = self.llm.generate(
            prompt_strs,
            sampling_params,
            use_tqdm=False
        )
        return [
            [preprocess(o.text) for o in out.outputs]
            for out in outputs
        ]



# =========================
# vLLM helpers
# =========================

def build_prompt(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def vllm_chat(llm, tokenizer, prompts, n, temperature=1.0):
    prompt_strs = [build_prompt(tokenizer, p) for p in prompts]
    sampling_params = SamplingParams(
        temperature=temperature,
        n=n,
        max_tokens=512,
    )
    outputs = llm.generate(prompt_strs, sampling_params, use_tqdm=False)
    return [[preprocess(o.text) for o in out.outputs] for out in outputs]


# =========================
# Parquet loader
# =========================

def load_parquet(path: str, max_samples: int = None) -> List[Dict]:
    table = pq.read_table(path)
    data = table.to_pylist()
    if max_samples is not None:
        data = data[:max_samples]
    return data


# =========================
# Main
# =========================

def main():
    # -------- config --------
    parquet_path = "/data/shenyz/verl_mt/data/train/wmt_en2fi_1k.parquet"
    model_name = "/data/models/Qwen3-4B"

    TOP_KS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    N_postedit = TOP_KS[-1]
    N_postedit = 1
    max_samples = 20

    # ✅ 新增：输出路径
    output_path = "postedit_quality_scores.txt"

    # -------- load data --------
    samples = load_parquet(parquet_path, max_samples=max_samples)
    print(f"Loaded {len(samples)} samples from parquet.")

    # -------- init models --------
    llm_actor = LLMActor.remote(model_name, TP_SIZE)
    chrf = chrFpp()
    all_comet_inputs = []
    meta = []


    # =========================
    # Loop over samples
    # =========================
    f = open("x.txt", "w")
    for idx, sample in enumerate(samples):
        src_text = sample["extra_info"]["src"]
        reference = sample["reward_model"]["ground_truth"]
        src_lang_name = LANG_NAME[sample["extra_info"]["src_lang"]]
        tgt_lang_name = LANG_NAME[sample["extra_info"]["tgt_lang"]]

        print(f"\n{'='*80}")
        print(f"Sample {idx+1}")
        print(f"{'='*80}")

        # =========================
        # 1️⃣ Translation
        # =========================
        translation_prompt = qwen_chat_input(
            src_lang_name,
            tgt_lang_name,
            src_text
        )
        draft_translation = ray.get(
            llm_actor.chat.remote(
                prompts=[translation_prompt],
                n=1,
                temperature=1.0
            )
        )[0][0]

        print("Draft translation:")
        print(draft_translation)
        f.write(f"Draft translation: \n{draft_translation}\n")

        # =========================
        # 2️⃣ Post-edit rollout
        # =========================
        pe_prompt = [{
            "role": "user",
            "content": build_postedit_prompt(
                src_text,
                draft_translation,
                tgt_lang_name
            )
        }]

        post_edits = ray.get(
            llm_actor.chat.remote(
                prompts=[pe_prompt],
                n=N_postedit,
                temperature=1.0
            )
        )[0]

        print(f"Generated {len(post_edits)} post-edits.\n {post_edits[0]} \n")
        f.write(f"Post: \n{post_edits[0]}\n")

        # =========================
        # 3️⃣ COMET-KIWI
        # =========================
        comet_inputs = [
            {"src": src_text, "mt": pe, "ref": reference}
            for pe in post_edits
        ]
        all_comet_inputs.append(comet_inputs)
        meta.append({
            "translations": post_edits,
            "reference": reference
        })
    
    f.close()
    print("Killing vLLM actor...")
    ray.kill(llm_actor)
    del llm_actor
    torch.cuda.empty_cache()

    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, \
        "GPU 不足：至少需要 >0"
    num_comet = num_gpus
    comet_actors = [CometActor.remote() for _ in range(num_comet)]
    print(f"Started {num_comet} COMET actors")

    # ========================================================
    # 2️⃣ Parallel COMET scoring
    # ========================================================

    futures = []
    for i, inputs in enumerate(all_comet_inputs):
        actor = comet_actors[i % num_comet]
        futures.append(actor.predict.remote(inputs))

    all_comet_scores = ray.get(futures)
    with open(output_path, "w", encoding="utf-8") as fout:
        for i, scores in enumerate(all_comet_scores):
            translations = meta[i]["translations"]
            reference = meta[i]["reference"]

            quality_scores = [
                chrf([t], [reference]) + c
                for t, c in zip(translations, scores)
            ]

            row = []
            print(f"\n[Sample {i+1}] Top-K Average Quality")
            for k in TOP_KS:
                avg_q = sum(quality_scores[:k]) / k
                row.append(avg_q)
                print(f"Top-{k:4d}: {avg_q:.4f}")

            fout.write(str(row) + ",\n")
            fout.flush()


if __name__ == "__main__":
    main()

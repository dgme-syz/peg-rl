# ============================================================
# Ray + vLLM + COMET full pipeline (LLM as Ray Actor, kill after use)
# ============================================================

import re
import sacrebleu
import torch
import ray
import pyarrow.parquet as pq
from vllm import LLM, SamplingParams
from comet import download_model, load_from_checkpoint
from transformers import AutoTokenizer
from typing import List, Dict

# =========================
# Language mapping
# =========================

TP_SIZE = 4


LANG_NAME = {
    "en": "English", "fi": "Finnish", "zh": "Chinese",
    "cs": "Czech", "de": "German", "fr": "French",
    "et": "Estonian", "kk": "Kazakh", "tr": "Turkish",
}

# =========================
# Prompt builders
# =========================

def qwen_chat_input(src_lang_name, tgt_lang_name, src_text):
    user_input = (
        f"Translate the following text into {tgt_lang_name} "
        f"without additional explanations:\n\n"
        f"{src_text}\n\n"
    )
    return [{"role": "user", "content": user_input}]

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

# =========================
# Utils
# =========================

def load_parquet(path: str, max_samples: int = None) -> List[Dict]:
    table = pq.read_table(path)
    data = table.to_pylist()
    if max_samples:
        data = data[:max_samples]
    return data

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

# ============================================================
# Ray Actors
# ============================================================

@ray.remote(num_gpus=1)
class CometActor:
    def __init__(self):
        model_path = download_model("Unbabel/wmt23-cometkiwi-da-xl")
        self.model = load_from_checkpoint(model_path)
        self.model.eval()

    def predict(self, inputs, batch_size=64):
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
            gpu_memory_utilization=0.8,
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

# ============================================================
# Main
# ============================================================

def main():
    parquet_path = "/home/nfs06/shenyz/data/recheck/train/wmt_en2fi_1k.parquet"
    model_name = "/home/nfs05/model/Qwen3-4B"

    TOP_KS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    N_TRANSLATION = TOP_KS[-1]
    max_samples = 100

    output_path = "translation_quality_scores_v1.txt"

    # -------------------------
    # Load data
    # -------------------------
    samples = load_parquet(parquet_path, max_samples=max_samples)
    print(f"Loaded {len(samples)} samples")

    # -------------------------
    # Init Ray
    # -------------------------
    ray.init()


    # -------------------------
    # Start LLM Actor
    # -------------------------
    print("Starting vLLM actor...")
    llm_actor = LLMActor.remote(model_name, TP_SIZE)

    # -------------------------
    # Start COMET Actors (剩余 GPU)
    # -------------------------


    chrf = chrFpp()

    # ========================================================
    # 1️⃣ Translation sampling (Ray LLM)
    # ========================================================

    all_comet_inputs = []
    meta = []

    for idx, sample in enumerate(samples):
        src_text = sample["extra_info"]["src"]
        reference = sample["reward_model"]["ground_truth"]
        src_lang = LANG_NAME[sample["extra_info"]["src_lang"]]
        tgt_lang = LANG_NAME[sample["extra_info"]["tgt_lang"]]

        print(f"\n[Sample {idx+1}] Translation sampling")

        prompt = qwen_chat_input(src_lang, tgt_lang, src_text)
        translations = ray.get(
            llm_actor.chat.remote(
                prompts=[prompt],
                n=N_TRANSLATION,
                temperature=1.0
            )
        )[0]

        comet_inputs = [
            {"src": src_text, "mt": t, "ref": reference}
            for t in translations
        ]
        print(translations[0])

        all_comet_inputs.append(comet_inputs)
        meta.append({
            "translations": translations,
            "reference": reference
        })

    # ========================================================
    # 🔥 Kill vLLM immediately
    # ========================================================

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
        print(f"[NOW] {i}/{len(all_comet_inputs)}")
        actor = comet_actors[i % num_comet]
        futures.append(actor.predict.remote(inputs))

    all_comet_scores = ray.get(futures)

    # ========================================================
    # 3️⃣ Aggregation + Top-K
    # ========================================================

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

    # ========================================================
    # Shutdown Ray
    # ========================================================

    print("Shutting down Ray...")
    ray.shutdown()
    print("Done.")

# ============================================================

if __name__ == "__main__":
    main()

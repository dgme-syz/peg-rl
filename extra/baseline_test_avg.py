# ============================================================
# Ray + vLLM + COMET-KIWI
# GRPO baseline Monte Carlo estimation
# ============================================================

import re
import textwrap
import sacrebleu
import torch
import ray
import pyarrow.parquet as pq
from vllm import LLM, SamplingParams
from comet import download_model, load_from_checkpoint
from transformers import AutoTokenizer
from typing import List, Dict
import math
import os

# ============================================================
# Config
# ============================================================

TP_SIZE = 8

NUM_SAMPLES = 100
N_TRANSLATION = 1024
N_POSTEDIT = 8

TOP_KS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

PARQUET_PATH = "/data/shenyz/verl_mt/data/train/wmt_en2fi_1k.parquet"
MODEL_NAME = "/data/models/Qwen3-4B"
OUTPUT_PATH = "grpo_baseline_scores.txt"

LANG_NAME = {
    "en": "English", "fi": "Finnish", "zh": "Chinese",
    "cs": "Czech", "de": "German", "fr": "French",
    "et": "Estonian", "kk": "Kazakh", "tr": "Turkish",
}

# ============================================================
# Prompt builders
# ============================================================

def qwen_chat_input(src_lang_name, tgt_lang_name, src_text):
    return [{
        "role": "user",
        "content": (
            f"Translate the following text into {tgt_lang_name} "
            f"without additional explanations:\n\n{src_text}\n"
        )
    }]

def build_postedit_prompt(src_text, draft_translation, target_lang_name):
    return textwrap.dedent(f"""
        Given the source text:

        {src_text}

        Improve the following draft {target_lang_name} translation
        into a high-quality {target_lang_name} version,
        without explanations:

        {draft_translation}
    """).strip()

# ============================================================
# Output preprocess
# ============================================================

def preprocess(text: str) -> str:
    parts = re.split(r'</think\s*>', text, flags=re.IGNORECASE)
    text = parts[-1]
    m = re.search(r'<text\s*>(.*?)</text\s*>', text,
                  flags=re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else text.strip()

# ============================================================
# Metrics
# ============================================================

class chrFpp:
    def __call__(self, responses, references):
        return sacrebleu.corpus_chrf(
            responses,
            [references],
            word_order=2,
            beta=2
        ).score / 100.0

# ============================================================
# Ray Actors
# ============================================================

@ray.remote(num_gpus=TP_SIZE)
class LLMActor:
    def __init__(self, model_name: str, tp_size: int):
        self.llm = LLM(
            model=model_name,
            tokenizer=model_name,
            trust_remote_code=True,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=0.9,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def chat(self, prompts, n, temperature=1.0):
        prompt_strs = [
            self.tokenizer.apply_chat_template(
                p,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for p in prompts
        ]
        outputs = self.llm.generate(
            prompt_strs,
            SamplingParams(
                temperature=temperature,
                n=n,
                max_tokens=512,
            ),
            use_tqdm=False
        )
        return [[preprocess(o.text) for o in out.outputs] for out in outputs]

@ray.remote(num_gpus=1)
class CometActor:
    def __init__(self):
        model_path = download_model("Unbabel/wmt23-cometkiwi-da-xl")
        self.model = load_from_checkpoint(model_path)
        self.model.eval()

    def predict(self, inputs, batch_size=128):
        with torch.no_grad():
            out = self.model.predict(
                inputs,
                batch_size=batch_size,
                gpus=1
            )
        return out["scores"]

# ============================================================
# Utils
# ============================================================

def load_parquet(path: str, max_samples: int):
    data = pq.read_table(path).to_pylist()
    return data[:max_samples]

# ============================================================
# Main
# ============================================================

def main():
    ray.init(ignore_reinit_error=True)

    samples = load_parquet(PARQUET_PATH, NUM_SAMPLES)
    print(f"Loaded {len(samples)} samples")

    llm = LLMActor.remote(MODEL_NAME, TP_SIZE)
    chrf = chrFpp()

    # --------------------------------------------------------
    # Translation + Post-edit rollout
    # --------------------------------------------------------

    all_comet_inputs = []
    meta = []

    for i, sample in enumerate(samples):
        print(f"\n===== Sample {i+1}/{len(samples)} =====")

        src = sample["extra_info"]["src"]
        ref = sample["reward_model"]["ground_truth"]
        src_lang = LANG_NAME[sample["extra_info"]["src_lang"]]
        tgt_lang = LANG_NAME[sample["extra_info"]["tgt_lang"]]

        # 1️⃣ translation rollout
        drafts = ray.get(
            llm.chat.remote(
                [qwen_chat_input(src_lang, tgt_lang, src)],
                n=N_TRANSLATION,
                temperature=1.0
            )
        )[0]

        # 2️⃣ post-edit rollout
        pe_prompts = [[{
            "role": "user",
            "content": build_postedit_prompt(src, d, tgt_lang)
        }] for d in drafts]

        post_edits = ray.get(
            llm.chat.remote(
                pe_prompts,
                n=N_POSTEDIT,
                temperature=1.0
            )
        )

        # flatten comet inputs
        comet_inputs = []
        for j in range(N_TRANSLATION):
            for pe in post_edits[j]:
                comet_inputs.append({
                    "src": src,
                    "mt": pe,
                    "ref": ref
                })

        all_comet_inputs.append(comet_inputs)
        meta.append({
            "post_edits": post_edits,
            "reference": ref
        })

    ray.kill(llm)
    torch.cuda.empty_cache()

    # --------------------------------------------------------
    # COMET scoring (parallel)
    # --------------------------------------------------------

    num_gpus = torch.cuda.device_count()
    comet_actors = [CometActor.remote() for _ in range(num_gpus)]

    futures = []
    for i, inputs in enumerate(all_comet_inputs):
        actor = comet_actors[i % num_gpus]
        futures.append(actor.predict.remote(inputs))

    all_comet_scores = ray.get(futures)

    # --------------------------------------------------------
    # Baseline estimation
    # --------------------------------------------------------

    with open(OUTPUT_PATH, "w") as f:
        for i in range(len(samples)):
            post_edits = meta[i]["post_edits"]
            ref = meta[i]["reference"]

            kiwi = torch.tensor(all_comet_scores[i]) \
                       .view(N_TRANSLATION, N_POSTEDIT)

            Q = []
            for j in range(N_TRANSLATION):
                chrf_scores = [
                    chrf([post_edits[j][k]], [ref])
                    for k in range(N_POSTEDIT)
                ]
                Q_j = (sum(chrf_scores) + kiwi[j].sum().item()) / N_POSTEDIT
                Q.append(Q_j)

            row = []
            print(f"\n[Sample {i+1}] GRPO baseline")
            for K in TOP_KS:
                b = sum(Q[:K]) / K
                row.append(b)
                print(f"Top-{K:4d}: {b:.4f}")

            f.write(str(row) + "\n")
            f.flush()

    print("Done.")

# ============================================================

if __name__ == "__main__":
    main()

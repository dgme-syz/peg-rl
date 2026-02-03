import re
import textwrap
import sacrebleu
import torch
from vllm import LLM, SamplingParams
from comet import download_model, load_from_checkpoint
from transformers import AutoTokenizer
from typing import List, Dict, Tuple


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


def build_postedit_prompt(src_text, pred_text, target_lang, LANG_DICT):
    return textwrap.dedent(
        f"""
        Given the source text:

        {src_text}

        Improve the following draft {LANG_DICT.get(target_lang, target_lang)} translation
        into a high-quality {LANG_DICT.get(target_lang, target_lang)} version,
        without explanations:

        {pred_text}
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
        flags=re.IGNORECASE | re.DOTALL,
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
            beta=2,
        ).score / 100.0


def load_comet_kiwi():
    model_path = download_model("Unbabel/wmt23-cometkiwi-da-xl")
    return load_from_checkpoint(model_path)


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

    return [
        [preprocess(o.text) for o in out.outputs]
        for out in outputs
    ]


# =========================
# Main
# =========================

def main():
    # -------- config --------
    # model_name = "/data/models/Qwen3-4B"
    model_name = "/data/shenyz/verl_mt/bs@128_n@72_m@1_@20251225_180735_@Qwen3-4B-mix_en_fi_mt-just-mix_@8gpus/global_step_100/actor/huggingface"
    # model_name = "/data/shenyz/verl_mt/bs@128_n@8_m@1_@20251226_124336_@Qwen3-4B-mix_en_fi_nofeedback_@8gpus/global_step_50/actor/huggingface"
    # model_name = "/data/shenyz/verl_mt/bs@128_n@8_m@1_@20251226_124336_@Qwen3-4B-mix_en_fi_nofeedback_@8gpus/global_step_105/actor/huggingface"

    N = 8   # translation rollouts
    M = 8   # post-edit rollouts
    comet_batch_size = 8

    src_text = "\"She had a real fear of food waste,\" Mr. Coe said."
    reference = "”Hän todellakin pelkäsi ruoan tuhlaamista”, Coe sanoi."

    src_lang_name = "English"
    tgt_lang_name = "Finnish"
    target_lang = "fi"
    LANG_DICT = {"fi": "Finnish"}

    comet = load_comet_kiwi()
    # -------- models --------
    llm = LLM(
        model=model_name,
        tokenizer=model_name,
        trust_remote_code=True,
        gpu_memory_utilization=0.5,
        tensor_parallel_size=2,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    chrf = chrFpp()

    # =========================
    # Translation rollout
    # =========================

    trans_prompt = qwen_chat_input(
        src_lang_name,
        tgt_lang_name,
        src_text,
    )

    translations = vllm_chat(
        llm,
        tokenizer,
        prompts=[trans_prompt],
        n=N,
        temperature=1.0,
    )[0]

    trajectories = []
    comet_inputs = []
    comet_index = []

    # ---- translations ----
    for i, trans in enumerate(translations):
        trajectories.append({
            "translation": {
                "text": trans,
                "chrfpp": chrf([trans], [reference]),
                "comet": None,
                "reward": None,
            },
            "post_edits": [],
            "stats": {},
        })

        comet_inputs.append({
            "src": src_text,
            "mt": trans,
            "ref": reference,
        })
        comet_index.append(("trans", i))

    # ---- post-edits ----
    for i, trans in enumerate(translations):
        pe_prompt = build_postedit_prompt(
            src_text,
            trans,
            target_lang,
            LANG_DICT,
        )

        pe_outputs = vllm_chat(
            llm,
            tokenizer,
            prompts=[[{"role": "user", "content": pe_prompt}]],
            n=M,
            temperature=1.0,
        )[0]

        for j, pe in enumerate(pe_outputs):
            trajectories[i]["post_edits"].append({
                "text": pe,
                "chrfpp": chrf([pe], [reference]),
                "comet": None,
                "reward": None,
            })

            comet_inputs.append({
                "src": src_text,
                "mt": pe,
                "ref": reference,
            })
            comet_index.append(("pe", i, j))

    # =========================
    # Batched COMET scoring
    # =========================

    with torch.no_grad():
        comet_scores = comet.predict(
            comet_inputs,
            batch_size=comet_batch_size,
            gpus=1,
        )["scores"]

    # ---- fill back scores & reward ----
    ptr = 0
    for tag in comet_index:
        score = comet_scores[ptr]
        ptr += 1

        if tag[0] == "trans":
            _, i = tag
            t = trajectories[i]["translation"]
            t["comet"] = score
            t["reward"] = t["chrfpp"] + score
        else:
            _, i, j = tag
            pe = trajectories[i]["post_edits"][j]
            pe["comet"] = score
            pe["reward"] = pe["chrfpp"] + score

    # =========================
    # Aggregate statistics
    # =========================

    for traj in trajectories:
        trans_r = traj["translation"]["reward"]
        pe_rewards = [pe["reward"] for pe in traj["post_edits"]]

        traj["stats"] = {
            "mean_post_edit_reward": sum(pe_rewards) / len(pe_rewards),
            "translation_reward": trans_r,
            "translation_plus_mean_pe": trans_r + sum(pe_rewards) / len(pe_rewards),
        }

    # =========================
    # Pretty print
    # =========================

    print("\n" + "=" * 80)
    print("SOURCE:")
    print(src_text)
    print("REFERENCE:")
    print(reference)
    print("=" * 80)

    for i, traj in enumerate(trajectories):
        t = traj["translation"]
        s = traj["stats"]

        print(f"\n[Translation #{i}]")
        print(
            f"  chrF++={t['chrfpp']:.2f} | COMET={t['comet']:.4f} | "
            f"R_trans={t['reward']:.4f}"
        )
        print(f"  mean R_post_edit={s['mean_post_edit_reward']:.4f}")
        print(f"  R_trans + mean R_pe = {s['translation_plus_mean_pe']:.4f}")
        print("  TEXT:")
        print(f"  {t['text']}")

        for j, pe in enumerate(traj["post_edits"]):
            print(
                f"\n    [Post-edit #{i}-{j}] "
                f"chrF++={pe['chrfpp']:.2f} | COMET={pe['comet']:.4f} | "
                f"R={pe['reward']:.4f}"
            )
            print(f"    {pe['text']}")

    print("\nDone.")


if __name__ == "__main__":
    main()

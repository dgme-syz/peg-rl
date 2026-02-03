import re
from typing import Optional, Tuple, List
import sacrebleu

# -----------------------
# Logger setup
# -----------------------
class PrintLogger:
    def info(self, *args, **kwargs): 
        print(*args, **kwargs)

    def warning(self, *args, **kwargs): 
        print("[WARNING]", *args, **kwargs)

    def error(self, *args, **kwargs): 
        print("[ERROR]", *args, **kwargs)

    def critical(self, *args, **kwargs): 
        print("[CRITICAL]", *args, **kwargs)

class NullLogger:
    def info(self, *args, **kwargs): pass
    def warning(self, *args, **kwargs): pass
    def error(self, *args, **kwargs): pass
    def critical(self, *args, **kwargs): pass

logger = PrintLogger()
# -----------------------
# Constants
# -----------------------
AGGLUTINATIVE_LANGS = {
    "tr", "fi", "et", "hu", "uz", "az", "ba", "tt", "ja", "ko", "cs"
}

# -----------------------
# Helper functions
# -----------------------
def extract_solution(text: str) -> str:
    processed = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return processed.strip()


def validate_response_structure(response: str, logger_obj=logger) -> bool:
    logger_obj.info("\n" + "=" * 60)
    logger_obj.info(" Structure Validation ".center(60, "="))

    tags = {"think_start": "<think>", "think_end": "</think>"}
    validation_passed = True
    positions = {}

    for name, tag in tags.items():
        count = response.count(tag)
        pos = response.find(tag)
        positions[name] = pos
        status = "OK" if count == 1 else "ERROR"
        logger_obj.info(f"  {tag:12} | count={count:2} | position={pos:4} | {status}")
        if count != 1:
            validation_passed = False

    if any(positions[k] == -1 for k in positions):
        logger_obj.info("  [Error] Missing required tags.")
        return False

    if not (positions["think_start"] < positions["think_end"]):
        logger_obj.info("  [Error] Invalid tag sequence: <think> must come before </think>")
        validation_passed = False
    else:
        logger_obj.info("  Tag sequence order: OK")

    logger_obj.info("=" * 60 + "\n")
    return validation_passed


def compute_metric(lang_pair: str, reference: str, prediction: str) -> Tuple[float, str]:
    prediction = prediction if isinstance(prediction, str) else ""
    src_lang, tgt_lang = lang_pair.split("-")

    if tgt_lang in AGGLUTINATIVE_LANGS:
        chrfpp = sacrebleu.corpus_chrf([prediction], [[reference]], word_order=2)
        return chrfpp.score, "chrF++"

    tokenize = "zh" if tgt_lang == "zh" else "ja-mecab" if tgt_lang == "ja" else "13a"
    bleu = sacrebleu.corpus_bleu([prediction], [[reference]], tokenize=tokenize)
    return bleu.score, "BLEU"


# -----------------------
# Compute score functions
# -----------------------
def compute_score_val_bleu(
    solution_str: str,
    ground_truth: str,
    lang_pair: str,
    print_ok: bool = True
) -> float:
    """Compute BLEU score for validation samples (without reward logic)."""
    logger_obj = logger if print_ok else NullLogger()

    if solution_str == "null":
        return -1

    logger_obj.info("\n" + "=" * 80)
    logger_obj.info(" Processing Validation Sample ".center(80, "="))

    answer_text = extract_solution(solution_str)
    logger_obj.info(f"\n[Prompt + Response]\n{solution_str}")

    score, mode = compute_metric(lang_pair, ground_truth, answer_text or "")
    logger_obj.info(f"Reference: {ground_truth}")
    logger_obj.info(f"Hypothesis: {answer_text}")
    logger_obj.info("\n" + "-" * 80)
    logger_obj.info(f"{mode} Score: {score}")

    return score


def compute_score(
    reward_metric: str,
    metric_scores: Optional[float] | List[float],
    lang_pair: str,
    solution_str: str,
    prompt_str: str,
    ground_truth: str,
    translation_raw: str,
    mul_times: int = 2,
    scale_factor: float = 100.0,
    thinking_check: bool = True,
    print_ok: bool = False,
    resp_token_length: Optional[int] = None,
    max_response_length: Optional[int] = None,
    score_limit: int = 95,
) -> float:
    """Compute total reward score based on model output and ground truth."""

    if max_response_length is None:
        raise ValueError("max_response_length is None, check mt_args.yaml")

    logger_obj = logger if print_ok else NullLogger()

    logger_obj.info("\n" + "=" * 80)
    logger_obj.info(" Processing Training Sample ".center(80, "="))

    format_correct = validate_response_structure(solution_str, logger_obj) if thinking_check else True
    answer_text = extract_solution(solution_str)

    logger_obj.info("\n" + "-" * 60)
    logger_obj.info(" Sample Info ".center(60, "-"))
    if translation_raw is None:
        logger_obj.info("[INFO] Stage 1 sample (no raw translation available).")
    else:
        min_score = min(metric_scores) if metric_scores else 0.0
        if answer_text == translation_raw and min_score < score_limit:
            logger_obj.info(f"[Answer]        {answer_text}")
            logger_obj.info(f"[Last Output]   {translation_raw}")
            logger_obj.info(f"[Min Comet]     {min_score}")
            logger_obj.info(f"[Answer==Raw]   {answer_text == translation_raw}")
            logger_obj.info(f"[Length]        answer={len(answer_text)}, raw={len(translation_raw)}")
            logger_obj.info("[INFO] Low-quality duplicate, giving 0.0 score.")
            return 0.0
    logger_obj.info("-" * 60 + "\n")

    logger_obj.info(" Prompt + Model Output ".center(80, "-"))
    logger_obj.info(prompt_str + solution_str)
    logger_obj.info("-" * 80 + "\n")

    if not (format_correct and answer_text and resp_token_length < max_response_length):
        logger_obj.info("[Content Validation] Skipped due to format error or empty answer.")
        logger_obj.info(
            f"[ERROR]-----details\n"
            f"[prompt_str + solution_str]: {prompt_str + solution_str}\n"
            f"[response]: {solution_str}\n"
            f"[response length]: {resp_token_length}\n"
            f"[max response length]: {max_response_length}\n"
        )
        return -1.0

    sentence_score, mode = compute_metric(lang_pair, ground_truth, answer_text)
    answer_score = 0.0

    def scale(val: float) -> float:
        return val / scale_factor

    if reward_metric == "Sentence":
        answer_score = scale(sentence_score)
    elif reward_metric == "Model":
        if metric_scores is None:
            raise ValueError("metric_scores is None; expected model score.")
        answer_score = scale(sum(metric_scores))
    elif reward_metric == "Merge":
        if metric_scores is None:
            raise ValueError("metric_scores is None; expected model score.")
        answer_score = scale(sentence_score + sum(metric_scores))
    else:
        raise ValueError(f"Invalid reward_metric type: {reward_metric}")

    logger_obj.info("\n" + "=" * 60)
    logger_obj.info(" Content Validation ".center(60, "="))
    logger_obj.info(f"Reference      : {ground_truth}")
    logger_obj.info(f"Hypothesis     : {answer_text}")
    logger_obj.info(f"{mode}           : {sentence_score:.2f}")
    if metric_scores is not None:
        logger_obj.info(f"Model Score(s) : {metric_scores}")
    logger_obj.info("=" * 60)

    total_score = pow(answer_score, mul_times) if mul_times != 1 else answer_score

    logger_obj.info("\n" + "-" * 60)
    logger_obj.info(" Reward Breakdown ".center(60, "-"))
    logger_obj.info(f"  Answer Score : {answer_score:.4f}")
    logger_obj.info(f"  Total Score  : {total_score:.4f}")
    logger_obj.info("-" * 60 + "\n")

    return total_score

def compute_score_corpus(
    solution_str: list[str],
    ground_truth: list[str],
    lang_pair: str,
    print_ok: bool = True
) -> Tuple[float, str]:
    logger_obj = logger if print_ok else NullLogger()

    assert len(solution_str) == len(ground_truth), (
        "The number of translations should be equal to the number of references"
    )

    _, tgt_lang = lang_pair.split("-")

    solution_str = [extract_solution(x) for x in solution_str]
    if any(x.startswith("<think>") for x in solution_str):
        logger_obj.info("Warning: Some responses still contain <think> tags after extraction.")
        c = 0
        for i in range(len(solution_str)):
            if solution_str[i].startswith("<think>"):
                solution_str[i] = "null"  # set null for fairness
                c += 1
        print(f"We have {c} null samples")

    if tgt_lang in AGGLUTINATIVE_LANGS:
        logger_obj.info(f"[Info] Target language={tgt_lang} detected as agglutinative. Using chrF++ instead of BLEU.")

        chrfpp = sacrebleu.corpus_chrf(
            solution_str,
            [ground_truth],
            word_order=2  # chrF++
        )
        return chrfpp.score, "chrF++"

    # 设置 tokenizer
    if tgt_lang == "zh":
        tokenizer = "zh"
    elif tgt_lang == "ja":
        tokenizer = "ja-mecab"
    elif tgt_lang == "ko":
        tokenizer = "ko-mecab"
    else:
        tokenizer = "13a"

    logger_obj.info(
        f"Preview of responses and references:\n"
        f"Response: {solution_str[0]}\nReference: {ground_truth[0]}"
    )


    bleu = sacrebleu.corpus_bleu(
        solution_str,
        [ground_truth],
        tokenize=tokenizer,
        force=True,
    )

    return bleu.score, "BLEU"

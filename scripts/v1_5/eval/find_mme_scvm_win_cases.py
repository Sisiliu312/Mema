#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find MME cases where scvm is correct but base is wrong."
    )
    parser.add_argument(
        "--base-jsonl",
        type=str,
        default="/dataset/eval/MME/answers/llava-v1.5-7b.jsonl",
        help="Path to base model jsonl.",
    )
    parser.add_argument(
        "--scvm-jsonl",
        type=str,
        default="/dataset/eval/MME/answers/llava-v1.5-7b-scvm.jsonl",
        help="Path to scvm model jsonl.",
    )
    parser.add_argument(
        "--mme-benchmark-dir",
        type=str,
        default="/dataset/eval/MME/MME_Benchmark_release_version/MME_Benchmark",
        help="Path to MME_Benchmark folder containing category data.",
    )
    parser.add_argument(
        "--out-jsonl",
        type=str,
        default="/dataset/eval/MME/answers/scvm_correct_base_wrong.jsonl",
        help="Output jsonl path.",
    )
    parser.add_argument(
        "--no-copy-images",
        action="store_true",
        help="Do not copy matched case images (default: copy to --image-out-dir).",
    )
    parser.add_argument(
        "--image-out-dir",
        type=str,
        default="/dataset/eval/MME/answers/scvm_correct_base_wrong_images",
        help="Directory to save copied images.",
    )
    return parser.parse_args()


def parse_pred_ans(pred_ans: str) -> str:
    pred_ans = pred_ans.strip().lower()
    if pred_ans in ("yes", "no"):
        return pred_ans
    prefix = pred_ans[:4]
    if "yes" in prefix:
        return "yes"
    if "no" in prefix:
        return "no"
    return "other"


def normalize_prompt_for_gt(prompt: str) -> str:
    p = prompt
    suffix = "Answer the question using a single word or phrase."
    if suffix in p:
        p = p.replace(suffix, "").strip()
    if "Please answer yes or no." not in p:
        p = p + " Please answer yes or no."
    return p


def get_gt(data_path: str) -> Dict[Tuple[str, str, str], str]:
    gt = {}
    for category in os.listdir(data_path):
        category_dir = os.path.join(data_path, category)
        if not os.path.isdir(category_dir):
            continue
        if os.path.exists(os.path.join(category_dir, "images")):
            qa_path = os.path.join(category_dir, "questions_answers_YN")
        else:
            qa_path = category_dir
        if not os.path.isdir(qa_path):
            continue
        for file_name in os.listdir(qa_path):
            if not file_name.endswith(".txt"):
                continue
            txt_path = os.path.join(qa_path, file_name)
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    question, answer = line.strip().split("\t")
                    gt[(category, file_name, question)] = answer.strip().lower()
    return gt


def load_answers(jsonl_path: str) -> List[dict]:
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))
    return items


def build_pred_map(answers: List[dict], gt: Dict[Tuple[str, str, str], str]) -> Dict[Tuple[str, str, str], dict]:
    pred_map = {}
    for ans in answers:
        qid = ans["question_id"]
        category = qid.split("/")[0]
        file_name = qid.split("/")[-1].split(".")[0] + ".txt"
        prompt = normalize_prompt_for_gt(ans["prompt"])

        key = (category, file_name, prompt)
        if key not in gt:
            # keep compatibility with convert_answer_to_mme.py fallback
            alt_prompt = prompt.replace(" Please answer yes or no.", "  Please answer yes or no.")
            key = (category, file_name, alt_prompt)
            if key not in gt:
                continue

        pred_map[key] = {
            "question_id": ans["question_id"],
            "prompt": ans["prompt"],
            "raw_pred": ans["text"],
            "pred_label": parse_pred_ans(ans["text"]),
        }
    return pred_map


def copy_case_images(win_cases: List[dict], mme_benchmark_dir: str, image_out_dir: str):
    benchmark_root = Path(mme_benchmark_dir)
    out_dir = Path(image_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing: List[str] = []
    seen = set()

    for item in win_cases:
        qid = item["question_id"]
        if "/" not in qid:
            missing.append(qid)
            continue
        category, file_name = qid.split("/", 1)

        # question_id may be like "artwork/10881.jpg" or "artwork/images/10881.jpg"
        src1 = benchmark_root / category / "images" / file_name
        src2 = benchmark_root / category / file_name
        src = src1 if src1.exists() else src2
        if not src.exists():
            missing.append(qid)
            continue

        real_src = str(src.resolve())
        if real_src in seen:
            continue
        seen.add(real_src)

        safe_rel_name = file_name.replace("/", "__")
        dst = out_dir / f"{category}__{safe_rel_name}"
        shutil.copy2(src, dst)
        copied += 1

    print(f"Copied images: {copied}")
    print(f"Missing images: {len(missing)}")
    print(f"Image output dir: {out_dir}")
    if missing:
        print("Missing question_id list:")
        for x in missing:
            print(f"  {x}")


def main():
    args = parse_args()
    gt = get_gt(args.mme_benchmark_dir)

    base_answers = load_answers(args.base_jsonl)
    scvm_answers = load_answers(args.scvm_jsonl)

    base_map = build_pred_map(base_answers, gt)
    scvm_map = build_pred_map(scvm_answers, gt)

    common_keys = sorted(set(base_map.keys()) & set(scvm_map.keys()))
    win_cases = []

    for key in common_keys:
        gt_label = gt[key]
        base_pred = base_map[key]["pred_label"]
        scvm_pred = scvm_map[key]["pred_label"]

        base_correct = base_pred == gt_label
        scvm_correct = scvm_pred == gt_label

        if scvm_correct and not base_correct:
            category, file_name, _ = key
            win_cases.append(
                {
                    "category": category,
                    "file": file_name,
                    "question_id": scvm_map[key]["question_id"],
                    "prompt": scvm_map[key]["prompt"],
                    "gt": gt_label,
                    "base_raw": base_map[key]["raw_pred"],
                    "base_label": base_pred,
                    "scvm_raw": scvm_map[key]["raw_pred"],
                    "scvm_label": scvm_pred,
                }
            )

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for item in win_cases:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Common comparable questions: {len(common_keys)}")
    print(f"scvm correct & base wrong: {len(win_cases)}")
    print(f"Saved to: {args.out_jsonl}")

    # Also print brief category distribution
    cate_cnt: Dict[str, int] = {}
    for x in win_cases:
        cate_cnt[x["category"]] = cate_cnt.get(x["category"], 0) + 1
    if cate_cnt:
        print("Category breakdown:")
        for c, n in sorted(cate_cnt.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {c}: {n}")

    if not args.no_copy_images:
        copy_case_images(win_cases, args.mme_benchmark_dir, args.image_out_dir)


if __name__ == "__main__":
    main()

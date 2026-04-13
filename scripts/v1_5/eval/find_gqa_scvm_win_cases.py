#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find GQA cases where scvm is correct but base is wrong."
    )
    parser.add_argument(
        "--questions-file",
        type=str,
        default="/dataset/eval/gqa/testdev_balanced_questions.json",
        help="GQA questions file used by eval.py.",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/dataset/eval/gqa/answers/llava_gqa_testdev_balanced/llava-v1.5-7b",
        help="Base answer directory containing merge.jsonl or chunk jsonl files.",
    )
    parser.add_argument(
        "--scvm-dir",
        type=str,
        default="/dataset/eval/gqa/answers/llava_gqa_testdev_balanced/llava-v1.5-7b-scvm",
        help="SCVM answer directory containing merge.jsonl or chunk jsonl files.",
    )
    parser.add_argument(
        "--out-jsonl",
        type=str,
        default="/dataset/eval/gqa/answers/scvm_correct_base_wrong.jsonl",
        help="Output cases jsonl path.",
    )
    parser.add_argument(
        "--image-out-dir",
        type=str,
        default="/dataset/eval/gqa/answers/scvm_correct_base_wrong_images",
        help="Directory for copied images.",
    )
    parser.add_argument(
        "--image-roots",
        nargs="+",
        default=[
            "/dataset/eval/gqa/images",
            "/home/ly/dataset/eval/gqa/images",
        ],
        help="Candidate roots to find GQA images.",
    )
    parser.add_argument(
        "--no-copy-images",
        action="store_true",
        help="Disable image copy step.",
    )
    return parser.parse_args()


def load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_answer_dir(answer_dir: str) -> Dict[str, dict]:
    """
    Support both:
    - merge.jsonl
    - chunk files like 2_0.jsonl, 2_1.jsonl, ...
    """
    d = Path(answer_dir)
    if not d.exists():
        raise FileNotFoundError(f"Answer dir not found: {answer_dir}")

    merge_file = d / "merge.jsonl"
    if merge_file.exists():
        rows = load_jsonl(str(merge_file))
    else:
        chunk_files = sorted(
            [
                p
                for p in d.glob("*.jsonl")
                if p.name != "merge.jsonl"
            ]
        )
        if not chunk_files:
            raise FileNotFoundError(f"No jsonl files found in: {answer_dir}")
        rows = []
        for p in chunk_files:
            rows.extend(load_jsonl(str(p)))

    ans_map: Dict[str, dict] = {}
    for row in rows:
        qid = str(row["question_id"])
        ans_map[qid] = row
    return ans_map


def normalize_prediction(text: str) -> str:
    # Keep same behavior as scripts/convert_gqa_for_eval.py
    return text.rstrip(".").lower()


def load_questions(path: str) -> Dict[str, dict]:
    # Official eval.py loads it with json.load and iterates dict items.
    with open(path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    return {str(k): v for k, v in questions.items()}


def find_image(image_name: str, roots: List[str]) -> Optional[Path]:
    if not image_name:
        return None
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        p = root_path / image_name
        if p.exists() and p.is_file():
            return p
        candidates = list(root_path.rglob(image_name))
        if candidates:
            for c in candidates:
                if c.is_file():
                    return c
    return None


def copy_case_images(cases: List[dict], image_roots: List[str], image_out_dir: str):
    out_dir = Path(image_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    seen = set()
    missing = []
    for item in cases:
        image_name = item["image"]
        src = find_image(image_name, image_roots)
        if src is None:
            missing.append(image_name)
            continue
        src_key = str(src.resolve())
        if src_key in seen:
            continue
        seen.add(src_key)
        dst = out_dir / image_name
        shutil.copy2(src, dst)
        copied += 1

    print(f"Copied images: {copied}")
    print(f"Missing images: {len(set(missing))}")
    print(f"Image output dir: {out_dir}")
    if missing:
        print("Missing image names:")
        for x in sorted(set(missing)):
            print(f"  {x}")


def main():
    args = parse_args()
    questions = load_questions(args.questions_file)
    base_map = load_answer_dir(args.base_dir)
    scvm_map = load_answer_dir(args.scvm_dir)

    common_qids = sorted(set(base_map.keys()) & set(scvm_map.keys()))
    cases = []
    skipped_not_balanced = 0
    missing_question = 0

    for qid in common_qids:
        q = questions.get(qid)
        if q is None:
            missing_question += 1
            continue
        if not q.get("isBalanced", False):
            skipped_not_balanced += 1
            continue

        gold = str(q["answer"]).lower()
        base_pred = normalize_prediction(str(base_map[qid]["text"]))
        scvm_pred = normalize_prediction(str(scvm_map[qid]["text"]))

        base_correct = base_pred == gold
        scvm_correct = scvm_pred == gold
        if scvm_correct and not base_correct:
            image_name = ""
            if "image" in q and q["image"]:
                image_name = str(q["image"])
            elif "imageId" in q and q["imageId"]:
                image_name = f"{q['imageId']}.jpg"
            cases.append(
                {
                    "question_id": qid,
                    "image": image_name,
                    "question": q.get("question", ""),
                    "gold": gold,
                    "base_raw": base_map[qid]["text"],
                    "base_pred_norm": base_pred,
                    "scvm_raw": scvm_map[qid]["text"],
                    "scvm_pred_norm": scvm_pred,
                }
            )

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for item in cases:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Common comparable questions: {len(common_qids)}")
    print(f"Missing question metadata: {missing_question}")
    print(f"Skipped non-balanced questions: {skipped_not_balanced}")
    print(f"scvm correct & base wrong: {len(cases)}")
    print(f"Saved to: {args.out_jsonl}")

    if not args.no_copy_images:
        copy_case_images(cases, args.image_roots, args.image_out_dir)


if __name__ == "__main__":
    main()

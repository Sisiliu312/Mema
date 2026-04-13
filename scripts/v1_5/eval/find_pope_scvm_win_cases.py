#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find POPE cases where scvm is correct but base is wrong."
    )
    parser.add_argument(
        "--annotation-dir",
        type=str,
        default="/dataset/eval/pope/coco",
        help="Directory containing coco_pope_*.json label files.",
    )
    parser.add_argument(
        "--question-file",
        type=str,
        default="/dataset/eval/pope/llava_pope_test.jsonl",
        help="POPE question file with image/category per question_id.",
    )
    parser.add_argument(
        "--base-jsonl",
        type=str,
        default="/dataset/eval/pope/answers/llava-v1.5-7b.jsonl",
        help="Base model answers jsonl.",
    )
    parser.add_argument(
        "--scvm-jsonl",
        type=str,
        default="/dataset/eval/pope/answers/llava-v1.5-7b-scvm.jsonl",
        help="SCVM model answers jsonl.",
    )
    parser.add_argument(
        "--out-jsonl",
        type=str,
        default="/dataset/eval/pope/answers/scvm_correct_base_wrong.jsonl",
        help="Output cases jsonl path.",
    )
    parser.add_argument(
        "--image-out-dir",
        type=str,
        default="/dataset/eval/pope/answers/scvm_correct_base_wrong_images",
        help="Directory for copied images.",
    )
    parser.add_argument(
        "--image-roots",
        nargs="+",
        default=[
            "/dataset/coco/val2014",
            "/dataset/eval/pope/val2014",
            "/dataset/eval/pope/images",
            "/dataset/coco/val2014",
            "/dataset/eval/pope/val2014",
            "/dataset/eval/pope/images",
        ],
        help="Candidate directories to find POPE image files.",
    )
    parser.add_argument(
        "--no-copy-images",
        action="store_true",
        help="Disable image copy step.",
    )
    return parser.parse_args()


def pope_pred_label(text: str) -> str:
    # Keep exactly the same normalization logic as eval_pope.py
    if text.find(".") != -1:
        text = text.split(".")[0]
    text = text.replace(",", "")
    words = text.split(" ")
    if "No" in words or "not" in words or "no" in words:
        return "no"
    return "yes"


def load_jsonl(path: str) -> List[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_questions(question_file: str) -> Dict[int, dict]:
    questions = {}
    for q in load_jsonl(question_file):
        questions[int(q["question_id"])] = q
    return questions


def load_labels(annotation_dir: str) -> Dict[str, Dict[int, str]]:
    labels_by_cat: Dict[str, Dict[int, str]] = {}
    for name in os.listdir(annotation_dir):
        if not (name.startswith("coco_pope_") and name.endswith(".json")):
            continue
        category = name[10:-5]
        label_map: Dict[int, str] = {}
        path = os.path.join(annotation_dir, name)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                label_map[int(row["question_id"])] = row["label"].strip().lower()
        labels_by_cat[category] = label_map
    return labels_by_cat


def build_answer_map(path: str) -> Dict[int, dict]:
    ans_map = {}
    for row in load_jsonl(path):
        ans_map[int(row["question_id"])] = row
    return ans_map


def find_image(image_name: str, roots: List[str]) -> Optional[Path]:
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        p = root_path / image_name
        if p.exists():
            return p
        # fallback for nested layouts
        candidates = list(root_path.rglob(image_name))
        if candidates:
            return candidates[0]
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
    questions = load_questions(args.question_file)
    labels_by_cat = load_labels(args.annotation_dir)
    base_map = build_answer_map(args.base_jsonl)
    scvm_map = build_answer_map(args.scvm_jsonl)

    common_qids = sorted(set(base_map.keys()) & set(scvm_map.keys()))
    cases = []
    missing_meta = 0
    for qid in common_qids:
        q = questions.get(qid)
        if q is None:
            missing_meta += 1
            continue
        category = q["category"]
        gt_label = labels_by_cat.get(category, {}).get(qid)
        if gt_label not in ("yes", "no"):
            missing_meta += 1
            continue

        base_raw = base_map[qid]["text"]
        scvm_raw = scvm_map[qid]["text"]
        base_pred = pope_pred_label(base_raw)
        scvm_pred = pope_pred_label(scvm_raw)

        base_correct = base_pred == gt_label
        scvm_correct = scvm_pred == gt_label
        if scvm_correct and not base_correct:
            cases.append(
                {
                    "question_id": qid,
                    "category": category,
                    "image": q["image"],
                    "prompt": base_map[qid].get("prompt", q.get("text", "")),
                    "gt_label": gt_label,
                    "base_raw": base_raw,
                    "base_pred": base_pred,
                    "scvm_raw": scvm_raw,
                    "scvm_pred": scvm_pred,
                }
            )

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for item in cases:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Common comparable questions: {len(common_qids)}")
    print(f"Missing question/label metadata: {missing_meta}")
    print(f"scvm correct & base wrong: {len(cases)}")
    print(f"Saved to: {args.out_jsonl}")

    # print category stats
    cate_cnt: Dict[str, int] = {}
    for x in cases:
        cate = x["category"]
        cate_cnt[cate] = cate_cnt.get(cate, 0) + 1
    if cate_cnt:
        print("Category breakdown:")
        for cate, n in sorted(cate_cnt.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {cate}: {n}")

    if not args.no_copy_images:
        copy_case_images(cases, args.image_roots, args.image_out_dir)


if __name__ == "__main__":
    main()

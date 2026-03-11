"""
使用官方 MP-DocVQA 指标（ANLS、Accuracy）评测已有预测结果。

适用于 LLaVA 等模型生成的 jsonl 预测文件，无需加载模型，仅需标注文件 + 预测文件。

用法:
  python eval_from_predictions.py \
    --annotation-file /path/to/MP-DocVQA/val.json \
    --result-file /path/to/answers/llava-v1.5-7b.jsonl

  # 或评测目录下所有 .jsonl
  python eval_from_predictions.py \
    --annotation-file /path/to/MP-DocVQA/val.json \
    --result-dir /path/to/answers/

标注文件格式: MP-DocVQA val.json，含 data 列表，每项有 questionId、answers。
预测文件格式: 每行一个 JSON，含 question_id、text（模型答案）。
"""
import os
import sys
import json
import argparse
import numpy as np

# 保证能导入同目录的 metrics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import Evaluator


def get_args():
    p = argparse.ArgumentParser(description="Evaluate MP-DocVQA from prediction jsonl (official metrics)")
    p.add_argument("--annotation-file", type=str, required=True,
                   help="Path to MP-DocVQA val.json")
    p.add_argument("--result-file", type=str, default=None,
                   help="Single result .jsonl file")
    p.add_argument("--result-dir", type=str, default=None,
                   help="Directory of .jsonl files to evaluate")
    p.add_argument("--case-sensitive", action="store_true", help="Case sensitive evaluation")
    return p.parse_args()


def load_annotations(annotation_file):
    with open(annotation_file, "r", encoding="utf-8") as f:
        ann = json.load(f)
    return {item["questionId"]: item for item in ann["data"]}


def load_predictions(result_file):
    results = []
    with open(result_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))
    return results


def evaluate_one(annotation_file, result_file, case_sensitive=False):
    annotations = load_annotations(annotation_file)
    results = load_predictions(result_file)

    gt_answers_list = []
    preds_list = []
    answer_types_list = []
    for r in results:
        qid = r.get("question_id")
        if qid not in annotations:
            continue
        gt = annotations[qid]
        if "answers" not in gt:
            continue
        gt_answers_list.append(gt["answers"])
        preds_list.append((r.get("text") or "").strip())
        # 标注里没有 answer_type 时默认 string
        answer_types_list.append(gt.get("answer_type", "string"))

    if not gt_answers_list:
        print(f"[WARN] No valid samples from {result_file}")
        return None

    evaluator = Evaluator(case_sensitive=case_sensitive)
    metrics = evaluator.get_metrics(gt_answers_list, preds_list, answer_types_list)
    mean_accuracy = np.mean(metrics["accuracy"])
    mean_anls = np.mean(metrics["anls"])
    return {
        "n_samples": len(gt_answers_list),
        "accuracy": mean_accuracy,
        "anls": mean_anls,
    }


def main():
    args = get_args()

    if args.result_file:
        name = os.path.splitext(os.path.basename(args.result_file))[0]
        out = evaluate_one(args.annotation_file, args.result_file, args.case_sensitive)
        if out:
            print(f"{name}: samples={out['n_samples']}, "
                  f"Accuracy={100.0 * out['accuracy']:.2f}%, ANLS={out['anls']:.4f}\n")

    if args.result_dir:
        for f in sorted(os.listdir(args.result_dir)):
            if not f.endswith(".jsonl"):
                continue
            path = os.path.join(args.result_dir, f)
            name = os.path.splitext(f)[0]
            out = evaluate_one(args.annotation_file, path, args.case_sensitive)
            if out:
                print(f"{name}: samples={out['n_samples']}, "
                      f"Accuracy={100.0 * out['accuracy']:.2f}%, ANLS={out['anls']:.4f}\n")


if __name__ == "__main__":
    main()

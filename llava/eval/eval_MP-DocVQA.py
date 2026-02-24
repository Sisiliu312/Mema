"""
MP-DocVQA 评估脚本。

用法:
  python -m llava.eval.eval_MP-DocVQA \
    --annotation-file /path/to/MP-DocVQA/val.json \
    --result-file /path/to/answer.jsonl

  # 或对目录下所有 .jsonl 结果依次评估
  python -m llava.eval.eval_MP-DocVQA \
    --annotation-file /path/to/MP-DocVQA/val.json \
    --result-dir /path/to/result_dir/

标注文件需为 MP-DocVQA 格式（含 data 列表，每项有 questionId、question、answers）。
结果文件为 LLaVA model_vqa 输出的 jsonl，每行含 question_id、prompt、text。
"""
import os
import argparse
import json

from llava.eval.m4c_evaluator import STVQAAccuracyEvaluator


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate MP-DocVQA results")
    parser.add_argument("--annotation-file", type=str, required=True,
                        help="Path to MP-DocVQA val.json (or train.json)")
    parser.add_argument("--result-file", type=str, default=None,
                        help="Path to single result .jsonl file")
    parser.add_argument("--result-dir", type=str, default=None,
                        help="Directory of result .jsonl files to evaluate")
    return parser.parse_args()


def eval_single(annotation_file, result_file):
    with open(annotation_file, "r", encoding="utf-8") as f:
        ann = json.load(f)
    annotations = {item["questionId"]: item for item in ann["data"]}

    results = [json.loads(line) for line in open(result_file, "r", encoding="utf-8")]

    pred_list = []
    for result in results:
        qid = result.get("question_id")
        if qid not in annotations:
            continue
        gt = annotations[qid]
        if "answers" not in gt:
            continue
        pred_list.append({
            "pred_answer": result.get("text", "").strip(),
            "gt_answers": gt["answers"],
        })

    if not pred_list:
        print(f"[WARN] No valid samples from {result_file}")
        return

    evaluator = STVQAAccuracyEvaluator()
    acc = evaluator.eval_pred_list(pred_list)
    name = os.path.splitext(os.path.basename(result_file))[0]
    print(f"{name}: samples={len(pred_list)}, Accuracy={100.0 * acc:.2f}%\n")


if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        eval_single(args.annotation_file, args.result_file)

    if args.result_dir is not None:
        for f in sorted(os.listdir(args.result_dir)):
            if not f.endswith(".jsonl"):
                continue
            eval_single(args.annotation_file, os.path.join(args.result_dir, f))

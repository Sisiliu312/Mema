#!/usr/bin/env python3
"""
ZeroBench 评估：读取 LLaVA 答案 jsonl 与 zerobench_gt.json，计算 pass@1 准确率。

官方推荐 prompt（与 HuggingFace/ZeroBench 一致）：
  prompt = question + "\\n\\nLet's think step by step and give the final answer in curly braces, like this: {final answer}"
评估时用 --mode official：从输出中取最后一个 {...} 内容，与 GT 做 strip+lower 后精确匹配且长度一致。
答案 jsonl 每行格式：{"question_id": "1", "prompt": "...", "text": "模型输出", ...}
GT json 格式：{"by_id": {"1": "11.90", ...}, "data": [...]}
"""
import os
import re
import json
import argparse


def normalize(s):
    """Strip + 合并空白，便于比较。"""
    if s is None:
        return ""
    s = " ".join(str(s).split()).strip()
    return s


def normalize_answer_zerobench(ans: str) -> str:
    """官方风格：strip、转小写、合并空白（与 ZeroBench 官方 snippet 一致）。"""
    if ans is None:
        return ""
    ans = ans.strip().lower()
    ans = re.sub(r"[\s]+", " ", ans)
    return ans.strip()


def extract_curly_answer(text: str) -> str:
    """
    从模型输出中抽取最后一对花括号内的内容，与官方一致：
    "like this: {final answer}" -> 取最后一个 {...} 的内容。
    """
    if not text:
        return ""
    formatted = text.strip().lower()
    try:
        # 官方: re.findall(pattern, formatted_response)[-1]
        pattern = r"\{(.*?)\}"
        matches = re.findall(pattern, formatted, re.DOTALL)
        if matches:
            return matches[-1].strip()
    except Exception:
        pass
    return ""


def extract_last_line(text):
    """取最后一行非空内容，常作为「最终答案」."""
    if not text:
        return ""
    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
    return lines[-1] if lines else ""


def extract_after_final(text):
    """尝试抽取 "final answer is X" / "answer is X" 等后的内容."""
    if not text:
        return ""
    text_lower = text.lower()
    for pattern in [
        r"final answer[:\s]+(?:is)?\s*[:\s]*([^\n.]+)",
        r"answer[:\s]+(?:is)?\s*[:\s]*([^\n.]+)",
        r"(?:therefore|thus),?\s*(?:the\s+)?(?:final\s+)?answer\s+is\s+([^\n.]+)",
    ]:
        m = re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL)
        if m:
            return normalize(m.group(1))
    return ""


def match_pred_gt(pred_raw, gt, mode="auto"):
    """
    判断 pred 是否与 gt 匹配。
    mode: official | exact | contain | last_line | auto
    - official: 与 ZeroBench 官方一致，从 {...} 抽取答案后精确匹配（strip+lower+同长）。
    """
    gt_n = normalize_answer_zerobench(gt)
    if not gt_n:
        return normalize(pred_raw or "") == ""

    if mode == "official":
        # 官方：抽取花括号内内容，再精确匹配（且长度一致，与官方逻辑等价）
        parsed = extract_curly_answer(pred_raw or "")
        parsed_n = normalize_answer_zerobench(parsed)
        # 官方: parsed_answer[:len(ground_truth)].lower() == ground_truth.strip().lower() and len(parsed_answer) == len(ground_truth.strip())
        gt_stripped = gt.strip().lower()
        parsed_stripped = parsed.strip().lower()
        return (
            parsed_stripped[: len(gt_stripped)] == gt_stripped
            and len(parsed_stripped) == len(gt_stripped)
        )

    pred_n = normalize(pred_raw or "")
    pred_n_zerobench = normalize_answer_zerobench(pred_raw or "")

    if mode == "exact":
        return pred_n_zerobench == gt_n

    if mode == "contain":
        return gt_n in pred_n_zerobench

    if mode == "last_line":
        last = normalize_answer_zerobench(extract_last_line(pred_raw or ""))
        return last == gt_n

    # auto: 先试 official（花括号），再 exact / contain / last_line / extract_after_final
    if match_pred_gt(pred_raw, gt, mode="official"):
        return True
    if pred_n_zerobench == gt_n:
        return True
    if gt_n in pred_n_zerobench:
        return True
    for extracted in [extract_last_line(pred_raw), extract_after_final(pred_raw)]:
        if normalize_answer_zerobench(extracted) == gt_n:
            return True
        if extracted and gt_n in normalize_answer_zerobench(extracted):
            return True
    return False


def eval_single(gt_path, result_path, mode="auto"):
    gt_data = json.load(open(gt_path, "r", encoding="utf-8"))
    by_id = gt_data.get("by_id", {})
    if not by_id and "data" in gt_data:
        by_id = {str(d["question_id"]): d["answer"] for d in gt_data["data"]}

    results = []
    with open(result_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))

    correct = 0
    total = 0
    missing_gt = []
    for r in results:
        qid = str(r.get("question_id", ""))
        pred = r.get("text", "")
        if qid not in by_id:
            missing_gt.append(qid)
            continue
        gt = by_id[qid]
        total += 1
        if match_pred_gt(pred, gt, mode=mode):
            correct += 1

    if missing_gt:
        print(f"  [WARN] No GT for question_ids: {missing_gt[:5]}{'...' if len(missing_gt) > 5 else ''}")
    acc = 100.0 * correct / total if total else 0.0
    return total, correct, acc


def main():
    parser = argparse.ArgumentParser(description="ZeroBench: evaluate LLaVA answer jsonl with GT json")
    parser.add_argument("--gt-json", type=str, required=True, help="zerobench_gt.json 路径")
    parser.add_argument("--result-file", type=str, default=None, help="答案 jsonl 文件")
    parser.add_argument("--result-dir", type=str, default=None, help="答案目录，对该目录下所有 .jsonl 评估")
    parser.add_argument("--mode", type=str, default="official",
                        choices=["official", "exact", "contain", "last_line", "auto"],
                        help="official=花括号抽取+精确匹配(与ZeroBench官方一致); exact/contain/last_line/auto=其他策略")
    args = parser.parse_args()

    if args.result_file:
        total, correct, acc = eval_single(args.gt_json, args.result_file, mode=args.mode)
        name = os.path.splitext(os.path.basename(args.result_file))[0]
        print(f"{name}: {correct}/{total} correct, Accuracy = {acc:.2f}%")

    if args.result_dir:
        for f in sorted(os.listdir(args.result_dir)):
            if not f.endswith(".jsonl"):
                continue
            path = os.path.join(args.result_dir, f)
            total, correct, acc = eval_single(args.gt_json, path, mode=args.mode)
            name = os.path.splitext(f)[0]
            print(f"{name}: {correct}/{total} correct, Accuracy = {acc:.2f}%")


if __name__ == "__main__":
    main()

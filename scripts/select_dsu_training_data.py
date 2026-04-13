"""
select_dsu_training_data.py
============================
从 llava_v1_5_mix665k.json 中挑选最适合训练 DSU + SpatialGate + AlignmentLoss 模块的 20k 样本。

选择逻辑：
  三个新模块对训练数据有不同侧重：
  - DSU：依赖有意义的 text_global（问题需对视觉内容有针对性）
  - SpatialGate：受益于空间丰富、多对象场景
  - AlignmentLoss：需要 answer_global 能代表视觉内容（答案要有描述性、有实质内容）

综合评分维度：
  1. 答案质量：答案足够长且非空洞（排除纯 yes/no、单词答案）
  2. 问题视觉针对性：问题中包含视觉推理关键词
  3. 对话丰富度：多轮对话覆盖图像的多个方面
  4. 数据来源多样性：按来源分层采样，避免 OCR/GQA 占比失衡

使用方法：
  python scripts/select_dsu_training_data.py \
      --input  /dataset/LLaVA-Tuning/llava_v1_5_mix665k.json \
      --output /dataset/LLaVA-Tuning/dsu_train_20k.json \
      --target 20000 \
      --seed   42
"""

import argparse
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path


# ─────────────────────────────────────────────
# 视觉推理关键词（用于评估问题的视觉针对性）
# ─────────────────────────────────────────────
VISUAL_KEYWORDS = re.compile(
    r"\b(describe|what|where|which|how many|how much|who|color|colour|"
    r"shape|size|position|location|left|right|above|below|behind|front|"
    r"between|near|far|count|number|scene|background|foreground|object|"
    r"person|people|animal|wearing|holding|doing|action|activity|"
    r"texture|material|type|kind|style|look|appear|see|show|find|"
    r"identify|explain|reason|relationship|difference|similar|compare|"
    r"spatial|direction|distance|arrangement|layout|setting)\b",
    re.IGNORECASE,
)

# 明显的"无效短答案"模式（对 AlignmentLoss 贡献极小）
SHORT_ANSWER_PATTERN = re.compile(
    r"^(yes|no|true|false|correct|incorrect|right|wrong|maybe|"
    r"unknown|none|n/a|not sure|i don'?t know|cannot determine)\.?$",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────
# 数据来源分层目标比例（基于可用量与质量权衡）
# ─────────────────────────────────────────────
SOURCE_QUOTA = {
    "coco":     0.45,   # 多轮、描述丰富，最适合 DSU
    "vg":       0.20,   # 空间关系/属性描述，适合 SpatialGate
    "gqa":      0.18,   # 组合推理，适合 DSU text 引导
    "textvqa":  0.10,   # OCR 理解，保留一定多样性
    "ocr_vqa":  0.07,   # OCR，少量保留
}
SOURCE_OTHER = 1.0 - sum(SOURCE_QUOTA.values())  # 剩余来源


def get_source(sample: dict) -> str:
    """根据 image 路径推断数据来源。"""
    img = sample.get("image", "")
    prefix = img.split("/")[0].lower()
    if prefix == "coco":
        return "coco"
    elif prefix in ("vg", "vg_100k", "vg_100k_2"):
        return "vg"
    elif prefix == "gqa":
        return "gqa"
    elif prefix == "textvqa":
        return "textvqa"
    elif prefix == "ocr_vqa":
        return "ocr_vqa"
    else:
        return "other"


def extract_turns(conversations: list):
    """分离 human / gpt 轮次，返回 (questions, answers)。"""
    questions = [c["value"] for c in conversations if c.get("from") == "human"]
    answers   = [c["value"] for c in conversations if c.get("from") == "gpt"]
    return questions, answers


def score_sample(sample: dict) -> float:
    """
    对单个样本打分，分数越高越适合 DSU/Gate/AlignmentLoss 训练。
    返回 [0, 1] 范围的综合分。
    """
    convs = sample.get("conversations", [])
    if not convs:
        return 0.0

    questions, answers = extract_turns(convs)
    if not answers:
        return 0.0

    # ── 1. 答案质量分（占 50%）──────────────────────────────────
    # 1a. 有效答案：非全空洞短答案
    valid_answers = [a for a in answers if not SHORT_ANSWER_PATTERN.match(a.strip())]
    valid_ratio = len(valid_answers) / len(answers)
    if valid_ratio == 0:
        return 0.0  # 全是 yes/no 答案，对 AlignmentLoss 无贡献

    # 1b. 总答案字符数（log 归一化，满分对应 ≥500 字符）
    total_ans_len = sum(len(a) for a in answers)
    ans_len_score = min(1.0, math.log1p(total_ans_len) / math.log1p(500))

    # 1c. 单轮平均答案长度（排除短答案带来的拖累）
    valid_ans_len = sum(len(a) for a in valid_answers)
    avg_valid_len = valid_ans_len / len(valid_answers) if valid_answers else 0
    avg_len_score = min(1.0, math.log1p(avg_valid_len) / math.log1p(150))

    ans_quality = 0.4 * valid_ratio + 0.35 * ans_len_score + 0.25 * avg_len_score

    # ── 2. 问题视觉针对性分（占 30%）────────────────────────────
    all_q_text = " ".join(questions)
    kw_hits = len(VISUAL_KEYWORDS.findall(all_q_text))
    # 命中关键词越多越好；≥5 个关键词给满分
    vis_score = min(1.0, kw_hits / 5.0)

    # ── 3. 对话丰富度分（占 20%）────────────────────────────────
    # 多轮对话能让 text_global 覆盖图像更多方面
    num_turns = len(answers)
    # 1轮:0.3, 2轮:0.6, 3轮:0.8, 4+轮:1.0
    turn_score = min(1.0, 0.3 + (num_turns - 1) * 0.25) if num_turns >= 1 else 0.0

    # ── 综合加权 ─────────────────────────────────────────────────
    score = 0.50 * ans_quality + 0.30 * vis_score + 0.20 * turn_score
    return score


def compute_min_score_per_source(
    samples_by_source: dict,
    quota_per_source: dict,
    margin: float = 1.5,
) -> dict:
    """
    对每个来源按分数降序，取 top-N（N = quota * margin），
    返回各来源的 score 阈值和候选列表。
    """
    result = {}
    for src, samples in samples_by_source.items():
        quota = quota_per_source.get(src, 0)
        if quota == 0:
            continue
        top_k = max(quota, int(quota * margin))
        ranked = sorted(samples, key=lambda x: x[1], reverse=True)[:top_k]
        result[src] = ranked
    return result


def select_samples(
    data: list,
    target: int = 20000,
    seed: int = 42,
) -> list:
    """核心选择函数：评分 → 按来源分层 → 最终采样。"""
    random.seed(seed)

    # ── Step 1: 过滤必须有 image 的样本并计算分数 ─────────────────
    print("Step 1/4: 过滤 + 打分…", flush=True)
    samples_by_source = defaultdict(list)
    skipped = 0

    for idx, sample in enumerate(data):
        if idx % 50000 == 0:
            print(f"  {idx}/{len(data)}", end="\r", flush=True)

        # 必须含图像
        if "image" not in sample or not sample["image"]:
            skipped += 1
            continue

        score = score_sample(sample)
        if score <= 0.0:
            skipped += 1
            continue

        src = get_source(sample)
        samples_by_source[src].append((sample, score))

    print(f"\n  有效样本: {sum(len(v) for v in samples_by_source.values())}, "
          f"跳过: {skipped}")
    for src, items in sorted(samples_by_source.items(), key=lambda x: -len(x[1])):
        print(f"  {src}: {len(items)}")

    # ── Step 2: 计算各来源目标数量 ───────────────────────────────
    print("Step 2/4: 计算分层配额…", flush=True)
    total_valid = sum(len(v) for v in samples_by_source.values())
    quota_per_source = {}

    for src, ratio in SOURCE_QUOTA.items():
        quota_per_source[src] = int(target * ratio)

    # "other" 来源分配剩余名额
    other_quota = target - sum(quota_per_source.values())
    quota_per_source["other"] = max(0, other_quota)

    # 若某来源实际可用量不足其配额，将缺口补给其他来源
    shortage = 0
    for src, quota in list(quota_per_source.items()):
        available = len(samples_by_source.get(src, []))
        if available < quota:
            shortage += quota - available
            quota_per_source[src] = available
    # 用 coco（最大来源）补足缺口
    if shortage > 0:
        coco_avail = len(samples_by_source.get("coco", []))
        extra = min(shortage, coco_avail - quota_per_source.get("coco", 0))
        quota_per_source["coco"] = quota_per_source.get("coco", 0) + extra

    for src, q in sorted(quota_per_source.items()):
        print(f"  {src}: quota={q}")

    # ── Step 3: 各来源取 top-K 候选，再随机 shuffle 采样 ─────────
    print("Step 3/4: 分层采样…", flush=True)
    selected = []
    for src, quota in quota_per_source.items():
        pool = samples_by_source.get(src, [])
        if not pool or quota == 0:
            continue
        # 取 top-(quota * 1.5) 高分样本，再随机打乱后截取 quota 个
        # 这样既保证质量又保留随机性
        top_k = min(len(pool), int(quota * 1.5))
        top_pool = sorted(pool, key=lambda x: x[1], reverse=True)[:top_k]
        random.shuffle(top_pool)
        chosen = top_pool[:quota]
        selected.extend(s for s, _ in chosen)
        print(f"  {src}: 从 top-{top_k} 中取 {len(chosen)} 条")

    # ── Step 4: 最终打乱 ──────────────────────────────────────────
    print("Step 4/4: 最终打乱…", flush=True)
    random.shuffle(selected)
    print(f"  最终选出 {len(selected)} 条样本")
    return selected


def print_statistics(selected: list):
    """打印选出样本的统计信息，便于验证质量。"""
    import statistics as stat

    print("\n──────── 选出样本统计 ────────")
    src_counter = defaultdict(int)
    all_ans_lens = []
    all_turns = []

    for s in selected:
        src_counter[get_source(s)] += 1
        _, answers = extract_turns(s["conversations"])
        total_len = sum(len(a) for a in answers)
        all_ans_lens.append(total_len)
        all_turns.append(len(answers))

    print("来源分布:")
    for src, cnt in sorted(src_counter.items(), key=lambda x: -x[1]):
        print(f"  {src}: {cnt} ({cnt/len(selected)*100:.1f}%)")

    print(f"\n答案总长度（chars）:")
    print(f"  均值: {stat.mean(all_ans_lens):.0f}")
    print(f"  中位数: {stat.median(all_ans_lens):.0f}")
    print(f"  p10: {sorted(all_ans_lens)[len(all_ans_lens)//10]}")
    print(f"  p90: {sorted(all_ans_lens)[int(len(all_ans_lens)*0.9)]}")

    print(f"\n对话轮数:")
    print(f"  均值: {stat.mean(all_turns):.2f}")
    print(f"  分布: " + str({k: all_turns.count(k) for k in sorted(set(all_turns))}))


def main():
    parser = argparse.ArgumentParser(description="为 DSU/Gate/AlignmentLoss 选择训练数据")
    parser.add_argument("--input",  default="/dataset/LLaVA-Tuning/llava_v1_5_mix665k.json")
    parser.add_argument("--output", default="/dataset/LLaVA-Tuning/dsu_train_20k.json")
    parser.add_argument("--target", type=int, default=20000)
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    # ── 加载 ─────────────────────────────────────────────────────
    print(f"加载 {args.input} …", flush=True)
    with open(args.input) as f:
        data = json.load(f)
    print(f"总共 {len(data)} 条样本")

    # ── 选择 ─────────────────────────────────────────────────────
    selected = select_samples(data, target=args.target, seed=args.seed)

    # ── 统计 ─────────────────────────────────────────────────────
    print_statistics(selected)

    # ── 保存 ─────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 已保存到 {out_path}  ({out_path.stat().st_size / 1024**2:.1f} MB)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Independent targeted audit of frozen sampling, leakage, predictions, and metrics."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def audit(root: Path) -> dict[str, Any]:
    root = root.resolve()
    data = root / "data"
    results = root / "results"
    rules = json.loads((data / "selection_rules.json").read_text(encoding="utf-8"))
    categories = rules["category_order"]
    tasks = rules["tasks"]
    selected = read_csv(data / "frozen_split.csv")
    candidates = read_csv(data / "weak_label_candidates.csv")
    cleaned = read_jsonl(data / "cleaned_january_articles.jsonl")
    unsup_manifest = read_csv(data / "unsupervised_corpus_manifest.csv")

    ids = [row["article_id"] for row in selected]
    check(len(ids) == len(set(ids)), "冻结样本含重复文章 ID")
    counts = Counter(row["label"] for row in selected)
    check(set(counts) == set(categories), f"冻结类别不完整: {dict(counts)}")
    target = int(rules["per_category_target"])
    if rules.get("selection_rules"):  # Reserve a compatibility hook without changing frozen rules.
        target = int(rules["selection_rules"].get("per_category_target", target))
    target = int(json.loads((data / "corpus_stats.json").read_text(encoding="utf-8"))["target_per_class"])
    split_target = rules["split_for_50"] if target == 50 else rules["split_for_40"]
    split_by_label: dict[str, Counter[str]] = defaultdict(Counter)
    selected_by_id = {row["article_id"]: row for row in selected}
    candidate_by_id = {row["article_id"]: row for row in candidates}
    for row in selected:
        label = row["label"]
        split_by_label[label][row["split"]] += 1
        check(row["article_id"] in candidate_by_id, f"被选文章不在冻结候选集: {row['article_id']}")
        check(candidate_by_id[row["article_id"]]["label"] == label, f"候选标签不一致: {row['article_id']}")
        expected_select = hashlib.sha256((rules["selection_salt"] + row["article_id"]).encode("utf-8")).hexdigest()
        expected_split = hashlib.sha256((rules["split_salt"] + row["article_id"]).encode("utf-8")).hexdigest()
        check(row["selection_hash"] == expected_select, f"选样哈希不一致: {row['article_id']}")
        check(row["split_hash"] == expected_split, f"切分哈希不一致: {row['article_id']}")
        article = next(item for item in cleaned if item["article_id"] == row["article_id"])
        check(int(row["body_token_count"]) >= int(rules["body_min_tokens"]), f"正文短于阈值: {row['article_id']}")
        check(int(row["input_token_count"]) <= int(rules["body_max_input_tokens"]), f"分类输入超过500词: {row['article_id']}")
        expected_text = " ".join(article["body_tokens"][: int(rules["body_max_input_tokens"])])
        check(row["body_text"] == expected_text, f"分类文本不是去标题后的正文前500词: {row['article_id']}")
    for label in categories:
        check(counts[label] == target, f"类别 {label} 样本数={counts[label]}，预期={target}")
        for split, n in split_target.items():
            check(split_by_label[label][split] == n, f"类别 {label} 的 {split} 数量={split_by_label[label][split]}，预期={n}")
        candidate_pool = sorted((row for row in candidates if row["label"] == label), key=lambda row: (row["selection_hash"], row["article_id"]))
        expected_ids = {row["article_id"] for row in candidate_pool[:target]}
        actual_ids = {row["article_id"] for row in selected if row["label"] == label}
        check(actual_ids == expected_ids, f"类别 {label} 不是按选样哈希取前 {target} 篇")
        split_order = sorted((row for row in selected if row["label"] == label), key=lambda row: (row["split_hash"], row["article_id"]))
        expected_splits = ["train"] * split_target["train"] + ["val"] * split_target["val"] + ["test"] * split_target["test"]
        check([row["split"] for row in split_order] == expected_splits, f"类别 {label} 的拆分不符合第二个哈希顺序")

    # Verify nested class tasks and their train/validation/test sizes.
    nested_counts: dict[str, dict[str, dict[str, int]]] = {}
    for task, labels in tasks.items():
        subset = [row for row in selected if row["label"] in labels]
        check(set(row["label"] for row in subset) == set(labels), f"{task} 子集类别不完整")
        nested_counts[task] = {split: {label: sum(row["label"] == label and row["split"] == split for row in subset) for label in labels} for split in ("train", "val", "test")}
        for label in labels:
            for split, n in split_target.items():
                check(nested_counts[task][split][label] == n, f"{task}/{label}/{split} 不满足嵌套切分")

    # Recompute all test/train/validation metrics from per-article prediction rows.
    predictions = read_csv(results / "predictions.csv")
    metric_rows = read_csv(results / "metrics.csv")
    metric_by_key = {(row["task"], row["method"], row["initialization"]): row for row in metric_rows}
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in predictions:
        grouped[(row["task"], row["method"], row["initialization"], row["split"])].append(row)
    recomputed_metric_rows = 0
    for (task, method, init, split), rows in grouped.items():
        check((task, method, init) in metric_by_key, f"预测缺少指标汇总行: {(task, method, init)}")
        labels = tasks[task]
        y_true = [row["true_label"] for row in rows]
        y_pred = [row["predicted_label"] for row in rows]
        accuracy = float(accuracy_score(y_true, y_pred))
        macro = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
        metric = metric_by_key[(task, method, init)]
        check(abs(float(metric[f"{split}_accuracy"]) - accuracy) < 1e-12, f"{task}/{method}/{init}/{split} accuracy 汇总不匹配")
        check(abs(float(metric[f"{split}_macro_f1"]) - macro) < 1e-12, f"{task}/{method}/{init}/{split} macro-F1 汇总不匹配")
        expected_n = sum(row["label"] in labels and row["split"] == split for row in selected)
        check(len(rows) == expected_n, f"{task}/{method}/{init}/{split} 预测数量={len(rows)}，预期={expected_n}")
        recomputed_metric_rows += 1

    six_key = ("6class", "official_fasttext", "pretrained", "test")
    six_rows = grouped[six_key]
    cm = confusion_matrix([row["true_label"] for row in six_rows], [row["predicted_label"] for row in six_rows], labels=categories)
    check(int(cm.sum()) == split_target["test"] * len(categories), "六类预训练测试混淆矩阵总数不等于测试样本数")
    saved_cm = read_csv(results / "confusion_matrix_6class_pretrained.csv")
    check(len(saved_cm) == len(categories), "六类混淆矩阵 CSV 行数错误")
    for i, row in enumerate(saved_cm):
        check(row["true_label"] == categories[i], "六类混淆矩阵行顺序错误")
        for j, label in enumerate(categories):
            check(int(row[label]) == int(cm[i, j]), f"六类混淆矩阵 CSV 计数错误: {categories[i]}->{label}")
    errors = read_csv(results / "errors_6class_pretrained_fasttext.csv")
    expected_errors = [row for row in six_rows if row["true_label"] != row["predicted_label"]]
    check(len(errors) == len(expected_errors), "六类错误 CSV 行数不匹配")
    check({row["article_id"] for row in errors} == {row["article_id"] for row in expected_errors}, "六类错误 CSV 文章 ID 不匹配")

    # Independently check validation/test isolation, including exact bodies and first 200 characters.
    article_by_id = {row["article_id"]: row for row in cleaned}
    eval_rows = [row for row in selected if row["split"] in {"val", "test"}]
    eval_ids = {row["article_id"] for row in eval_rows}
    eval_full = {digest(article_by_id[row["article_id"]]["body_text"]) for row in eval_rows}
    eval_prefix = {digest(article_by_id[row["article_id"]]["body_text"][:200]) for row in eval_rows}
    included = [row for row in unsup_manifest if row["included"].lower() == "true"]
    included_ids = {row["article_id"] for row in included}
    check(not (included_ids & eval_ids), "验证或测试文章直接进入词向量语料")
    for row in included:
        check(row["full_body_sha256"] not in eval_full, f"验证/测试正文重复进入词向量语料: {row['article_id']}")
        check(row["first_200_chars_sha256"] not in eval_prefix, f"验证/测试前200字符重复进入词向量语料: {row['article_id']}")
    reason_counts = Counter(row["reason"] for row in unsup_manifest if row["reason"] != "included")
    included_articles_jsonl = {row["article_id"] for row in cleaned if row["article_id"] in included_ids and row["body_tokens"]}
    check(included_ids == included_articles_jsonl, "词向量清单和有正文的清洗语料文章不一致")
    for row in unsup_manifest:
        article = article_by_id[row["article_id"]]
        full_hash = digest(article["body_text"])
        prefix_hash = digest(article["body_text"][:200])
        if row["article_id"] in eval_ids:
            expected_reason = "validation_or_test_id"
        elif full_hash in eval_full:
            expected_reason = "duplicate_full_body_of_validation_or_test"
        elif prefix_hash in eval_prefix:
            expected_reason = "duplicate_first_200_chars_of_validation_or_test"
        elif not article["body_tokens"]:
            expected_reason = "empty_body_no_tokens"
        else:
            expected_reason = "included"
        check(row["reason"] == expected_reason, f"词向量文章纳入/排除原因不匹配: {row['article_id']}")
    corpus_lines = (data / "unsupervised_corpus.txt").read_text(encoding="utf-8").splitlines()
    expected_corpus_lines = [" ".join(article_by_id[row["article_id"]]["body_tokens"]) for row in unsup_manifest if row["reason"] == "included"]
    check(corpus_lines == expected_corpus_lines, "词向量语料内容或文章顺序与冻结 manifest 不一致")

    report = {
        "status": "PASS",
        "selected_articles": len(selected),
        "class_counts": dict(counts),
        "split_counts_by_label": {label: dict(split_by_label[label]) for label in categories},
        "nested_task_split_counts": nested_counts,
        "metric_groups_recomputed": recomputed_metric_rows,
        "prediction_rows": len(predictions),
        "six_class_test_confusion_total": int(cm.sum()),
        "six_class_test_errors": len(errors),
        "unsupervised_included_articles": len(included),
        "unsupervised_excluded_reason_counts": dict(reason_counts),
        "validation_test_ids_absent_from_unsupervised": True,
        "validation_test_body_and_prefix_duplicates_absent": True,
    }
    (results / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    result = audit(root)
    print(json.dumps(result, ensure_ascii=False, indent=2))

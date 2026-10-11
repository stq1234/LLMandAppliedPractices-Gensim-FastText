#!/usr/bin/env python3
"""Independent targeted audit of frozen sampling, leakage, predictions, and metrics."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import jieba
import pyarrow.parquet as parquet
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_external_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    kept = [char for char in normalized if unicodedata.category(char) not in {"Cc", "Cf", "Cs"}]
    return re.sub(r"\s+", " ", "".join(kept)).strip()


def canonical_external_body(value: str) -> str:
    normalized = normalize_external_text(value).casefold()
    return "".join(
        char for char in normalized
        if not char.isspace() and not unicodedata.category(char).startswith("P")
    )


def is_punctuation_only(token: str) -> bool:
    return bool(token) and all(unicodedata.category(char).startswith("P") for char in token)


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

    # The classifier sample, keyword candidates, audit table, and rules are
    # immutable inputs. These hashes were recorded before the corpus switch.
    frozen_hashes = json.loads((data / "frozen_input_sha256.json").read_text(encoding="utf-8"))
    actual_frozen_hashes = {
        name: hashlib.sha256((data / name).read_bytes()).hexdigest()
        for name in frozen_hashes
    }
    check(actual_frozen_hashes == frozen_hashes, "冻结 PeopleDaily 分类输入摘要发生变化")
    source_lock = json.loads((data / "external_source_lock.json").read_text(encoding="utf-8"))
    check(source_lock["frozen_people_daily_sha256_before_change"] == frozen_hashes, "外部来源锁未记录原始冻结分类文件摘要")
    check(source_lock["dataset"] == "bltlab/lr-sum" and source_lock["config"] == "cmn_s" and source_lock["split"] == "train", "外部来源不是固定的 LR-Sum cmn_s train")
    check(source_lock["revision"] == "deb2e73ce53e42ceda31297a991afa138ee665a8", "LR-Sum revision 不符合冻结版本")
    check(source_lock["license"] == "CC BY 4.0", "LR-Sum 许可记录缺失")
    check(source_lock["embedding_input_column"] == "text", "表示学习输入列不是正文 text")
    check(source_lock["source_labels_used"] is False, "LR-Sum 来源标签被用于语料")
    check(source_lock["trace_column"] == "id", "LR-Sum 来源 ID 未单独标注为追踪信息")
    check({"title", "summary", "dataset labels"}.issubset(set(source_lock["excluded_columns_from_embedding_tokens"])), "来源锁未明确排除 LR-Sum 标题、摘要和标签")
    raw_path = root / "data" / "external" / "lr_sum_cmn_s_train.parquet"
    raw_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    check(raw_hash == "687db5968b23e492bfb904dc438e6ecff95f711ab93c8ef3d4d36e086d5e8630", "LR-Sum Parquet 摘要与固定 revision 不符")
    check(raw_hash == source_lock["source_sha256"], "LR-Sum Parquet 与来源锁摘要不匹配")
    raw_table = parquet.read_table(raw_path)
    check(set(raw_table.column_names) == {"id", "url", "title", "summary", "text"}, "LR-Sum Parquet schema 与来源记录不匹配")
    external_records = raw_table.select(["id", "text"]).to_pylist()
    external_manifest = read_csv(data / "unsupervised_corpus_manifest.csv")
    external_cleaned = read_jsonl(data / "lr_sum_cmn_s_cleaned_train.jsonl")

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
    expected_method_keys = {
        (task, method, initialization)
        for task in tasks
        for method, initialization in (
            ("tfidf_logreg", ""),
            ("w2v_mean_logreg", ""),
            ("fasttext_mean_logreg", ""),
            ("official_fasttext", "random"),
            ("official_fasttext", "pretrained"),
        )
    }
    check(len(metric_rows) == 15 and set(metric_by_key) == expected_method_keys, "分类指标不是预设的 3 任务 × 5 方法/初始化组合")
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
    expected_prediction_groups = {
        (*key, split) for key in expected_method_keys for split in ("train", "val", "test")
    }
    check(set(grouped) == expected_prediction_groups and recomputed_metric_rows == 45, "逐篇预测未覆盖 15 个方法组合的 train/val/test")
    expected_prediction_total = sum(
        5 * sum(row["label"] in tasks[task] for row in selected)
        for task in tasks
    )
    check(len(predictions) == expected_prediction_total, "逐篇预测总量与固定样本及 5 个模型不符")

    curves = read_csv(results / "fasttext_validation_curve.csv")
    check(len(curves) == 18, "fastText 验证曲线应覆盖 3 任务 × 2 初始化 × 3 个候选轮数")
    curve_by_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in curves:
        curve_by_key[(row["task"], row["initialization"])].append(row)
    for task in tasks:
        for initialization in ("random", "pretrained"):
            candidates_for_fit = curve_by_key[(task, initialization)]
            check({int(row["epoch"]) for row in candidates_for_fit} == {25, 100, 250}, f"{task}/{initialization} 候选轮数有缺漏")
            picked = min(candidates_for_fit, key=lambda row: (-float(row["val_macro_f1"]), int(row["epoch"])))
            metric = metric_by_key[(task, "official_fasttext", initialization)]
            check(int(metric["selected_epoch"]) == int(picked["epoch"]), f"{task}/{initialization} 不是依据验证 Macro-F1（平局取小轮数）选择 epoch")

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

    # Independently reconstruct the external text-only corpus, including
    # normalization, deduplication, tokenization, and cross-source overlap.
    article_by_id = {row["article_id"]: row for row in cleaned}
    eval_rows = [row for row in selected if row["split"] in {"val", "test"}]
    eval_ids = {row["article_id"] for row in eval_rows}
    pd_all_full: set[str] = set()
    pd_all_prefix: set[str] = set()
    pd_eval_full: set[str] = set()
    pd_eval_prefix: set[str] = set()
    for article in cleaned:
        canonical = canonical_external_body(article["body_text"])
        if not canonical:
            continue
        full = digest(canonical)
        prefix = digest(canonical[:200])
        pd_all_full.add(full)
        pd_all_prefix.add(prefix)
        if article["article_id"] in eval_ids:
            pd_eval_full.add(full)
            pd_eval_prefix.add(prefix)

    jieba.setLogLevel(20)
    expected_manifest: list[dict[str, Any]] = []
    expected_cleaned: list[dict[str, Any]] = []
    expected_corpus_lines: list[str] = []
    seen_external: set[str] = set()
    exact_any_pd = prefix_any_pd = exact_eval = prefix_eval = 0
    for row_index, source_row in enumerate(external_records):
        source_id = str(source_row.get("id") or "")
        normalized = normalize_external_text(source_row.get("text") or "")
        canonical = canonical_external_body(normalized)
        full = digest(canonical) if canonical else ""
        prefix = digest(canonical[:200]) if canonical else ""
        tokens: list[str] = []
        if not canonical:
            reason = "empty_body"
        elif full in seen_external:
            reason = "duplicate_exact_external"
        else:
            seen_external.add(full)
            if full in pd_all_full:
                exact_any_pd += 1
            elif prefix in pd_all_prefix:
                prefix_any_pd += 1
            if full in pd_eval_full:
                exact_eval += 1
                reason = "duplicate_exact_people_daily"
            elif prefix in pd_eval_prefix:
                prefix_eval += 1
                reason = "duplicate_prefix_200_people_daily"
            elif full in pd_all_full:
                reason = "duplicate_exact_people_daily"
            elif prefix in pd_all_prefix:
                reason = "duplicate_prefix_200_people_daily"
            else:
                tokens = [
                    token.strip()
                    for token in jieba.cut(normalized, cut_all=False)
                    if token.strip() and not is_punctuation_only(token.strip())
                ]
                reason = "included" if tokens else "empty_after_tokenization"
        expected_manifest.append({
            "source_row": row_index,
            "source_id": source_id,
            "normalized_text_sha256": digest(normalized) if normalized else "",
            "canonical_body_sha256": full,
            "canonical_prefix_200_sha256": prefix,
            "token_count": len(tokens),
            "included": reason == "included",
            "reason": reason,
            "text_only": True,
        })
        if reason == "included":
            expected_corpus_lines.append(" ".join(tokens))
            expected_cleaned.append({"source_row": row_index, "source_id": source_id, "normalized_text": normalized, "tokens": tokens})

    check(len(external_manifest) == len(expected_manifest), "外部语料 manifest 行数与 pinned parquet 不匹配")
    for index, (actual, expected) in enumerate(zip(external_manifest, expected_manifest)):
        for field, expected_value in expected.items():
            actual_value: Any = actual[field]
            if field == "source_row" or field == "token_count":
                actual_value = int(actual_value)
            elif field in {"included", "text_only"}:
                actual_value = actual_value.lower() == "true"
            check(actual_value == expected_value, f"LR-Sum manifest 第 {index} 行字段 {field} 不匹配")
    check(external_cleaned == expected_cleaned, "保存的 LR-Sum 清洗正文/token 与原始 text 列重建结果不一致")
    corpus_lines = (data / "unsupervised_corpus.txt").read_text(encoding="utf-8").splitlines()
    check(corpus_lines == expected_corpus_lines, "表示学习语料与 LR-Sum train text 重建结果不一致")
    included_external = sum(row["included"] for row in expected_manifest)
    reason_counts = Counter(row["reason"] for row in expected_manifest if row["reason"] != "included")
    token_total = sum(row["token_count"] for row in expected_manifest if row["included"])
    external_stats = json.loads((data / "external_corpus_stats.json").read_text(encoding="utf-8"))
    check(external_stats["source_rows"] == len(external_records) and external_stats["included_articles"] == included_external, "外部语料统计计数与实际清洗文件不匹配")
    check(external_stats["excluded_reasons"] == dict(sorted(reason_counts.items())), "外部语料排除原因计数不匹配")
    check(external_stats["included_tokens"] == token_total, "外部语料 token 总数不匹配")
    check(external_stats["cross_corpus_exact_matches_any_people_daily"] == exact_any_pd, "跨语料完全重复统计不匹配")
    check(external_stats["cross_corpus_prefix_200_matches_any_people_daily"] == prefix_any_pd, "跨语料前缀重复统计不匹配")
    check(external_stats["cross_corpus_exact_matches_validation_test"] == exact_eval, "评测集完全重复统计不匹配")
    check(external_stats["cross_corpus_prefix_200_matches_validation_test"] == prefix_eval, "评测集前缀重复统计不匹配")
    check(external_stats["title_used"] is False and external_stats["summary_used"] is False and external_stats["source_labels_used"] is False, "LR-Sum 标题、摘要或来源标签进入表示学习")
    check(external_stats["tokenizer"] == "jieba" and external_stats["jieba_version"] == "0.42.1", "外部语料 tokenizer 版本不符合冻结设置")
    check(exact_eval == 0 and prefix_eval == 0, "PeopleDaily 验证/测试正文与外部表示学习语料存在重复")
    embedding_meta = json.loads((results / "embedding_training.json").read_text(encoding="utf-8"))
    corpus_hash = hashlib.sha256((data / "unsupervised_corpus.txt").read_bytes()).hexdigest()
    check(embedding_meta["unsupervised_corpus_sha256"] == corpus_hash, "Gensim 训练元数据语料 SHA 与当前外部语料不匹配")
    check(embedding_meta["external_source_sha256"] == raw_hash and embedding_meta["external_revision"] == source_lock["revision"], "Gensim 训练元数据来源不是固定 LR-Sum revision")
    check(embedding_meta["w2v"]["corpus_count"] == included_external and embedding_meta["gensim_fasttext"]["corpus_count"] == included_external, "Gensim 模型训练文章计数与外部语料不符")
    check((root / embedding_meta["w2v"]["artifact"]["path"]).is_file() and (root / embedding_meta["gensim_fasttext"]["artifact"]["path"]).is_file(), "Gensim 模型文件缺失")
    from gensim.models import FastText, Word2Vec

    w2v_model = Word2Vec.load(str(root / embedding_meta["w2v"]["artifact"]["path"]))
    ft_model = FastText.load(str(root / embedding_meta["gensim_fasttext"]["artifact"]["path"]))
    for model_name, model, saved in (
        ("w2v", w2v_model, embedding_meta["w2v"]),
        ("gensim_fasttext", ft_model, embedding_meta["gensim_fasttext"]),
    ):
        params = saved["params"]
        check(model.corpus_count == included_external, f"{model_name} checkpoint corpus_count 不符")
        check(
            model.sg == params["sg"] == 1
            and model.vector_size == params["vector_size"] == 100
            and model.window == params["window"] == 5
            and model.min_count == params["min_count"] == 3
            and model.epochs == params["epochs"] == 10
            and model.workers == params["workers"] == 1
            and model.seed == params["seed"] == 42,
            f"{model_name} checkpoint 参数与元数据不符",
        )
        check(len(model.wv) == saved["vocabulary_size"], f"{model_name} checkpoint 词表大小与元数据不符")
    check(ft_model.wv.min_n == 2 and ft_model.wv.max_n == 4 and ft_model.wv.bucket == 100000, "Gensim FastText 子词参数不符")

    train_token_counts = Counter(
        token for row in selected if row["split"] == "train" for token in row["body_text"].split()
    )
    train_vocabulary = set(train_token_counts)
    coverage_csv = list(csv.DictReader((results / "train_vocabulary_coverage.csv").open(encoding="utf-8-sig")))
    coverage_by_model = {row["model"]: row for row in coverage_csv}
    for model_name, model in (("word2vec", w2v_model.wv), ("gensim_fasttext", ft_model.wv)):
        known = train_vocabulary.intersection(model.key_to_index)
        occurrences = sum(train_token_counts[token] for token in known)
        total_occurrences = sum(train_token_counts.values())
        saved_coverage = coverage_by_model.get(model_name, {})
        check(int(saved_coverage.get("train_vocabulary_types", -1)) == len(train_vocabulary), f"{model_name} 训练词表类型计数不符")
        check(int(saved_coverage.get("known_types", -1)) == len(known), f"{model_name} 已知词类型覆盖数不符")
        check(int(saved_coverage.get("train_token_occurrences", -1)) == total_occurrences, f"{model_name} 训练 token 频次计数不符")
        check(int(saved_coverage.get("known_token_occurrences", -1)) == occurrences, f"{model_name} 已知 token 频次覆盖数不符")
        expected_type_coverage = len(known) / len(train_vocabulary) if train_vocabulary else 0
        expected_token_coverage = occurrences / total_occurrences if total_occurrences else 0
        check(math.isclose(float(saved_coverage.get("type_coverage", -1)), expected_type_coverage, rel_tol=0, abs_tol=1e-12), f"{model_name} 类型覆盖率不符")
        check(math.isclose(float(saved_coverage.get("token_coverage", -1)), expected_token_coverage, rel_tol=0, abs_tol=1e-12), f"{model_name} 词频覆盖率不符")
        meta_coverage = next((row for row in embedding_meta.get("train_vocabulary_coverage", []) if row["model"] == model_name), {})
        check(int(meta_coverage.get("train_vocabulary_types", -1)) == len(train_vocabulary), f"{model_name} meta 训练词表类型计数不符")
        check(int(meta_coverage.get("known_types", -1)) == len(known), f"{model_name} meta 已知类型覆盖数不符")
        check(int(meta_coverage.get("train_token_occurrences", -1)) == total_occurrences, f"{model_name} meta token 频次计数不符")
        check(int(meta_coverage.get("known_token_occurrences", -1)) == occurrences, f"{model_name} meta 已知 token 频次覆盖数不符")
        check(math.isclose(float(meta_coverage.get("type_coverage", -1)), expected_type_coverage, rel_tol=0, abs_tol=1e-12), f"{model_name} meta 类型覆盖率不符")
        check(math.isclose(float(meta_coverage.get("token_coverage", -1)), expected_token_coverage, rel_tol=0, abs_tol=1e-12), f"{model_name} meta 词频覆盖率不符")

    vec_info = embedding_meta["pretrained_vectors"]
    vec_path = root / vec_info["path"]
    known_train_tokens = sorted(train_vocabulary.intersection(ft_model.wv.key_to_index))
    check(vec_path.is_file(), "监督训练词表 .vec 未生成")
    vec_tokens: list[str] = []
    zero_vectors = 0
    malformed_vectors = 0
    with vec_path.open("r", encoding="utf-8") as handle:
        header = handle.readline().split()
        check(header == [str(len(known_train_tokens)), "100"], "监督训练词表 .vec header 与 known train 交集不符")
        for line in handle:
            parts = line.rstrip("\n").split(" ")
            if len(parts) != 101:
                malformed_vectors += 1
                continue
            vec_tokens.append(parts[0])
            try:
                values = [float(value) for value in parts[1:]]
            except ValueError:
                malformed_vectors += 1
                continue
            if not all(math.isfinite(value) for value in values):
                malformed_vectors += 1
            elif not any(value != 0 for value in values):
                zero_vectors += 1
    check(vec_tokens == known_train_tokens, ".vec 导出词项不是恰好 train vocabulary ∩ FastText key_to_index 的排序结果")
    check(malformed_vectors == 0 and zero_vectors == 0, ".vec 含非有限、零向量或格式错误的行")
    check(vec_info["train_vocabulary_tokens"] == len(train_vocabulary), ".vec train vocabulary 类型数错误")
    check(vec_info["exported_known_tokens"] == len(known_train_tokens) and vec_info["oov_exported"] == 0, ".vec 导出数量或 OOV 说明错误")
    check(vec_path.stat().st_size == vec_info["bytes"], ".vec 文件大小与元数据不符")

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
        "frozen_classification_inputs_unchanged": True,
        "embedding_source": {
            "dataset": "bltlab/lr-sum",
            "config": "cmn_s",
            "split": "train",
            "revision": source_lock["revision"],
            "source_sha256": raw_hash,
            "source_rows": len(external_records),
            "included_articles": included_external,
            "included_tokens": token_total,
            "excluded_reason_counts": dict(reason_counts),
            "title_summary_labels_used": False,
            "cross_corpus_exact_matches_any_people_daily": exact_any_pd,
            "cross_corpus_prefix_200_matches_any_people_daily": prefix_any_pd,
            "cross_corpus_eval_exact_matches": exact_eval,
            "cross_corpus_eval_prefix_200_matches": prefix_eval,
        },
        "unsupervised_excluded_reason_counts": dict(reason_counts),
        "evaluation_text_absent_from_external_embeddings": True,
        "embedding_checkpoints_loaded_and_parameters_verified": True,
        "train_vocabulary_types": len(train_vocabulary),
        "fasttext_known_train_vocabulary_types": len(known_train_tokens),
        "pretrained_vec_rows": len(vec_tokens),
        "pretrained_vec_zero_vectors": zero_vectors,
    }
    (results / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    result = audit(root)
    print(json.dumps(result, ensure_ascii=False, indent=2))

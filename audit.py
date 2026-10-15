#!/usr/bin/env python3
"""Check frozen artifacts, source isolation, predictions, and metrics.

Teacher-corpus sample reconstruction deliberately reuses teacher_corpus.py, so
this is a result-consistency audit rather than an independent parser check.
"""

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

from external_corpus import EXPECTED_PARQUET_SHA256, RELATIVE_PARQUET
from teacher_corpus import build_teacher_corpus


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_text(value: str) -> str:
    return digest_bytes(value.encode("utf-8"))


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


def punctuation_only(token: str) -> bool:
    return bool(token) and all(unicodedata.category(char).startswith("P") for char in token)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _same_csv_rows(actual: list[dict[str, str]], expected: list[dict[str, Any]], fields: list[str], name: str) -> None:
    expected_text = [{field: str(row.get(field, "")) for field in fields} for row in expected]
    actual_text = [{field: row.get(field, "") for field in fields} for row in actual]
    check(actual_text == expected_text, f"{name} 与教师源重新生成的记录不一致")


def _weak_label_cue_overlap(selected: list[dict[str, str]], categories: list[str]) -> dict[str, Any]:
    """Recompute exact title weak-label cue overlap in the stored classifier input."""
    by_split: dict[str, dict[str, Any]] = {}
    by_split_and_label: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        split_rows = [row for row in selected if row["split"] == split]
        overlaps: list[tuple[dict[str, str], list[str]]] = []
        for row in split_rows:
            title_cues: dict[str, set[str]] = defaultdict(set)
            for entry in row["matched_keywords"].split(";"):
                if not entry:
                    continue
                category, separator, cue_text = entry.partition(":")
                check(bool(separator) and category in categories, f"{row['article_id']} 弱标签关键词格式错误")
                title_cues[category].update(cue for cue in cue_text.split("|") if cue)
            check(set(title_cues) == {row["label"]}, f"{row['article_id']} 没有唯一类别的标题关键词")
            body_tokens = set(row["body_text"].split())
            found = sorted(title_cues[row["label"]] & body_tokens)
            overlaps.append((row, found))

        hit_count = sum(bool(found) for _, found in overlaps)
        by_split[split] = {
            "articles": len(split_rows),
            "articles_with_any_title_label_keyword_in_classifier_body": hit_count,
            "fraction": hit_count / len(split_rows) if split_rows else None,
            "percent": 100 * hit_count / len(split_rows) if split_rows else None,
        }
        for label in categories:
            label_rows = [(row, found) for row, found in overlaps if row["label"] == label]
            if not label_rows:
                continue
            label_hits = sum(bool(found) for _, found in label_rows)
            by_split_and_label[f"{split}:{label}"] = {
                "articles": len(label_rows),
                "articles_with_any_title_label_keyword_in_classifier_body": label_hits,
                "fraction": label_hits / len(label_rows),
                "article_ids_with_overlap": [row["article_id"] for row, found in label_rows if found],
                "matched_keywords_found_in_body": {row["article_id"]: found for row, found in label_rows if found},
            }
    return {"by_split": by_split, "by_split_and_label": by_split_and_label}


def audit(root: Path) -> dict[str, Any]:
    root = root.resolve()
    data = root / "data"
    results = root / "results"
    prep = build_teacher_corpus(root, write_outputs=False)
    rules = prep["rules"]
    categories = rules["category_order"]
    tasks = rules["tasks"]
    selected = read_csv(data / "frozen_split.csv")
    split_fields = [
        "article_id", "label", "split", "selection_hash", "split_hash", "title",
        "body_token_count", "input_token_count", "body_text", "matched_keywords",
        "canonical_body_sha256", "source_members",
    ]
    _same_csv_rows(selected, prep["selected_rows"], split_fields, "冻结拆分")
    _same_csv_rows(
        read_csv(data / "weak_label_candidates.csv"), prep["candidate_rows"],
        ["article_id", "label", "source_members", "title", "body_token_count", "body_sha256", "selection_hash", "matched_keywords"],
        "弱标签候选表",
    )
    _same_csv_rows(
        read_csv(data / "label_filter_audit.csv"), prep["article_audit_rows"],
        ["article_id", "source_members", "segment_count", "body_token_count", "body_sha256", "title", "matched_categories", "matched_keywords", "status", "duplicate_of"],
        "筛选审计表",
    )
    _same_csv_rows(
        read_csv(data / "teacher_decode_errors.csv"), prep["decode_errors"],
        ["source_member", "line_number", "absolute_byte_offset", "article_id", "segment_id", "record_id", "invalid_byte_hex"],
        "GB18030 解码错误表",
    )
    cue_overlap = _weak_label_cue_overlap(selected, categories)
    saved_cue_overlap = json.loads((results / "weak_label_cue_overlap.json").read_text(encoding="utf-8"))
    check(saved_cue_overlap.get("source_file") == "data/frozen_split.csv", "弱标签线索统计来源错误")
    check(saved_cue_overlap.get("source_rows") == len(selected), "弱标签线索统计样本数错误")
    check(saved_cue_overlap.get("this_is_split_leakage") is False, "弱标签线索统计的解释标记错误")
    check(saved_cue_overlap.get("by_split") == cue_overlap["by_split"], "弱标签线索统计的 split 汇总不符")
    check(saved_cue_overlap.get("by_split_and_label") == cue_overlap["by_split_and_label"], "弱标签线索统计的逐类逐篇明细不符")
    check(json.loads((data / "corpus_stats.json").read_text(encoding="utf-8")) == prep["stats"], "教师语料统计与源文件重新计数不一致")
    check(json.loads((data / "teacher_source_lock.json").read_text(encoding="utf-8")) == prep["source_lock"], "教师语料来源锁不一致")
    check(json.loads((data / "selection_rules.json").read_text(encoding="utf-8")) == prep["rules"], "弱标签规则文件与运行配置不一致")

    target = int(rules["per_category_target"])
    split_target = rules["split_per_class"]
    counts = Counter(row["label"] for row in selected)
    split_by_label = {label: Counter(row["split"] for row in selected if row["label"] == label) for label in categories}
    check(counts == Counter({label: target for label in categories}), f"类别样本数不均衡：{dict(counts)}")
    check(all(split_by_label[label] == Counter(split_target) for label in categories), "类别内 train/val/test 数量不符合冻结比例")
    check(len({row["article_id"] for row in selected}) == len(selected), "冻结样本含重复文章 ID")
    check(all(int(row["input_token_count"]) <= rules["body_max_input_tokens"] for row in selected), "分类正文超过 500 token 限制")
    check(all(len(row["body_text"].split()) == int(row["input_token_count"]) for row in selected), "输入 token 数与正文不符")
    body_hashes_by_split = {
        split: {row["canonical_body_sha256"] for row in selected if row["split"] == split}
        for split in ("train", "val", "test")
    }
    check(not (body_hashes_by_split["train"] & body_hashes_by_split["val"]), "训练集和验证集正文指纹重叠")
    check(not (body_hashes_by_split["train"] & body_hashes_by_split["test"]), "训练集和测试集正文指纹重叠")
    check(not (body_hashes_by_split["val"] & body_hashes_by_split["test"]), "验证集和测试集正文指纹重叠")

    nested_counts: dict[str, dict[str, dict[str, int]]] = {}
    for task, labels in tasks.items():
        subset = [row for row in selected if row["label"] in labels]
        nested_counts[task] = {
            split: {label: sum(row["label"] == label and row["split"] == split for row in subset) for label in labels}
            for split in ("train", "val", "test")
        }
        check(len(subset) == target * len(labels), f"{task} 样本总数不符合嵌套子集")
        for label in labels:
            for split in ("train", "val", "test"):
                check(nested_counts[task][split][label] == split_target[split], f"{task}/{label}/{split} 数量错误")

    # Rebuild LR-Sum's text-only corpus and cross-check every valid, unique
    # teacher body. This verifies the same leak boundary used for pretraining.
    source_lock = json.loads((data / "external_source_lock.json").read_text(encoding="utf-8"))
    check(source_lock["classification_source"] == prep["source_lock"], "LR-Sum 来源锁没有绑定到教师分类语料")
    check(source_lock["dataset"] == "bltlab/lr-sum" and source_lock["config"] == "cmn_s" and source_lock["split"] == "train", "外部语料来源不是固定的 LR-Sum cmn_s train")
    check(source_lock["revision"] == "deb2e73ce53e42ceda31297a991afa138ee665a8", "LR-Sum revision 不符合冻结值")
    check(source_lock["license"] == "CC BY 4.0" and source_lock["embedding_input_column"] == "text", "LR-Sum 来源许可或正文列记录缺失")
    check(source_lock["source_labels_used"] is False and source_lock["trace_column"] == "id", "LR-Sum 标签或追踪 ID 隔离记录错误")
    raw_path = root / RELATIVE_PARQUET
    raw_bytes = raw_path.read_bytes()
    raw_hash = digest_bytes(raw_bytes)
    check(raw_hash == EXPECTED_PARQUET_SHA256 == source_lock["source_sha256"], "LR-Sum Parquet 摘要与固定 revision 不符")
    source_table = parquet.read_table(raw_path)
    check({"id", "title", "summary", "text"}.issubset(source_table.column_names), "LR-Sum Parquet 缺少预期列")
    external_records = source_table.select(["id", "text"]).to_pylist()
    teacher_full: set[str] = set()
    teacher_prefix: set[str] = set()
    teacher_eval_full: set[str] = set()
    teacher_eval_prefix: set[str] = set()
    eval_ids = {row["article_id"] for row in selected if row["split"] in {"val", "test"}}
    for article in prep["articles"]:
        canonical = canonical_external_body(article["body_text"])
        if not canonical:
            continue
        full_hash = digest_text(canonical)
        prefix_hash = digest_text(canonical[:200])
        teacher_full.add(full_hash)
        teacher_prefix.add(prefix_hash)
        if article["article_id"] in eval_ids:
            teacher_eval_full.add(full_hash)
            teacher_eval_prefix.add(prefix_hash)

    jieba.setLogLevel(20)
    manifest_path = data / "unsupervised_corpus_manifest.csv"
    actual_manifest = read_csv(manifest_path)
    expected_manifest: list[dict[str, Any]] = []
    expected_cleaned: list[dict[str, Any]] = []
    expected_corpus_lines: list[str] = []
    seen_external: set[str] = set()
    all_teacher_exact = all_teacher_prefix = eval_exact = eval_prefix = 0
    excluded_reasons: Counter[str] = Counter()
    token_total = 0
    for source_row_index, record in enumerate(external_records):
        source_id = str(record.get("id") or "")
        normalized = normalize_external_text(record.get("text") or "")
        canonical = canonical_external_body(normalized)
        full_hash = digest_text(canonical) if canonical else ""
        prefix_hash = digest_text(canonical[:200]) if canonical else ""
        tokens: list[str] = []
        if not canonical:
            reason = "empty_body"
        elif full_hash in seen_external:
            reason = "duplicate_exact_external"
        else:
            seen_external.add(full_hash)
            if full_hash in teacher_full:
                all_teacher_exact += 1
            elif prefix_hash in teacher_prefix:
                all_teacher_prefix += 1
            if full_hash in teacher_eval_full:
                eval_exact += 1
                reason = "duplicate_exact_teacher"
            elif prefix_hash in teacher_eval_prefix:
                eval_prefix += 1
                reason = "duplicate_prefix_200_teacher"
            elif full_hash in teacher_full:
                reason = "duplicate_exact_teacher"
            elif prefix_hash in teacher_prefix:
                reason = "duplicate_prefix_200_teacher"
            else:
                tokens = [token.strip() for token in jieba.cut(normalized, cut_all=False) if token.strip() and not punctuation_only(token.strip())]
                reason = "included" if tokens else "empty_after_tokenization"
        if reason != "included":
            excluded_reasons[reason] += 1
        expected_manifest.append({
            "source_row": source_row_index,
            "source_id": source_id,
            "normalized_text_sha256": digest_text(normalized) if normalized else "",
            "canonical_body_sha256": full_hash,
            "canonical_prefix_200_sha256": prefix_hash,
            "token_count": len(tokens),
            "included": reason == "included",
            "reason": reason,
            "text_only": True,
        })
        if reason == "included":
            token_total += len(tokens)
            expected_corpus_lines.append(" ".join(tokens))
            expected_cleaned.append({"source_row": source_row_index, "source_id": source_id, "normalized_text": normalized, "tokens": tokens})
    check(len(actual_manifest) == len(expected_manifest), "LR-Sum manifest 行数与原始 Parquet 不符")
    for index, (actual, expected) in enumerate(zip(actual_manifest, expected_manifest)):
        for field, expected_value in expected.items():
            actual_value: Any = actual[field]
            if field in {"source_row", "token_count"}:
                actual_value = int(actual_value)
            elif field in {"included", "text_only"}:
                actual_value = actual_value.lower() == "true"
            check(actual_value == expected_value, f"LR-Sum manifest 第 {index} 行字段 {field} 不符")
    cleaned_rows = [json.loads(line) for line in (data / "lr_sum_cmn_s_cleaned_train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    check(cleaned_rows == expected_cleaned, "已保存的 LR-Sum 清洗正文/token 与原始 text 列重建结果不一致")
    corpus_lines = (data / "unsupervised_corpus.txt").read_text(encoding="utf-8").splitlines()
    check(corpus_lines == expected_corpus_lines, "Gensim 语料与固定 LR-Sum train text 重建结果不一致")
    external_stats = json.loads((data / "external_corpus_stats.json").read_text(encoding="utf-8"))
    included_external = len(expected_cleaned)
    check(external_stats["source_rows"] == len(external_records), "LR-Sum 原始行数统计不符")
    check(external_stats["included_articles"] == included_external, "LR-Sum 纳入文章数不符")
    check(external_stats["excluded_reasons"] == dict(sorted(excluded_reasons.items())), "LR-Sum 排除原因统计不符")
    check(external_stats["included_tokens"] == token_total, "LR-Sum token 数统计不符")
    check(external_stats["cross_corpus_exact_matches_any_teacher"] == all_teacher_exact, "LR-Sum 与所有有效教师正文完全匹配计数不符")
    check(external_stats["cross_corpus_prefix_200_matches_any_teacher"] == all_teacher_prefix, "LR-Sum 与所有有效教师正文前缀匹配计数不符")
    check(external_stats["cross_corpus_exact_matches_validation_test"] == eval_exact, "LR-Sum 与验证/测试正文完全匹配计数不符")
    check(external_stats["cross_corpus_prefix_200_matches_validation_test"] == eval_prefix, "LR-Sum 与验证/测试正文前缀匹配计数不符")
    check(external_stats["title_used"] is False and external_stats["summary_used"] is False and external_stats["source_labels_used"] is False, "LR-Sum 非正文信息进入词向量语料")
    check(external_stats["tokenizer"] == "jieba" and external_stats["jieba_version"] == "0.42.1", "LR-Sum 分词器版本不符")
    check(eval_exact == 0 and eval_prefix == 0, "LR-Sum 表示学习语料包含验证/测试正文")

    embedding_meta = json.loads((results / "embedding_training.json").read_text(encoding="utf-8"))
    corpus_hash = digest_bytes((data / "unsupervised_corpus.txt").read_bytes())
    check(embedding_meta["unsupervised_corpus_sha256"] == corpus_hash, "Gensim checkpoint 对应的语料摘要不符")
    check(embedding_meta["external_source_sha256"] == raw_hash and embedding_meta["external_revision"] == source_lock["revision"], "Gensim checkpoint LR-Sum 来源不符")
    check(embedding_meta["w2v"]["corpus_count"] == included_external and embedding_meta["gensim_fasttext"]["corpus_count"] == included_external, "Gensim checkpoint 文章数不符")
    from gensim.models import FastText, Word2Vec

    w2v_model = Word2Vec.load(str(root / embedding_meta["w2v"]["artifact"]["path"]))
    ft_model = FastText.load(str(root / embedding_meta["gensim_fasttext"]["artifact"]["path"]))
    for name, model, saved in (("w2v", w2v_model, embedding_meta["w2v"]), ("gensim_fasttext", ft_model, embedding_meta["gensim_fasttext"])):
        params = saved["params"]
        check(model.corpus_count == included_external, f"{name} 词向量文章数不符")
        check(
            model.sg == params["sg"] == 1
            and model.vector_size == params["vector_size"] == 100
            and model.window == params["window"] == 5
            and model.min_count == params["min_count"] == 3
            and model.epochs == params["epochs"] == 10
            and model.workers == params["workers"] == 1
            and model.seed == params["seed"] == 42,
            f"{name} 模型参数不符",
        )
        check(len(model.wv) == saved["vocabulary_size"], f"{name} 词表大小不符")
    check(ft_model.wv.min_n == 2 and ft_model.wv.max_n == 4 and ft_model.wv.bucket == 100000, "Gensim FastText 字符 n-gram 参数不符")

    train_token_counts = Counter(token for row in selected if row["split"] == "train" for token in row["body_text"].split())
    train_vocabulary = set(train_token_counts)
    coverage_rows = read_csv(results / "train_vocabulary_coverage.csv")
    coverage_by_model = {row["model"]: row for row in coverage_rows}
    for model_name, keyed_vectors in (("word2vec", w2v_model.wv), ("gensim_fasttext", ft_model.wv)):
        known = train_vocabulary.intersection(keyed_vectors.key_to_index)
        occurrences = sum(train_token_counts[token] for token in known)
        total_occurrences = sum(train_token_counts.values())
        row = coverage_by_model[model_name]
        check(int(row["train_vocabulary_types"]) == len(train_vocabulary), f"{model_name} 训练词表类型计数不符")
        check(int(row["known_types"]) == len(known), f"{model_name} 已知词数不符")
        check(int(row["train_token_occurrences"]) == total_occurrences, f"{model_name} 训练 token 总数不符")
        check(int(row["known_token_occurrences"]) == occurrences, f"{model_name} 已知 token 数不符")
        check(math.isclose(float(row["type_coverage"]), len(known) / len(train_vocabulary), rel_tol=0, abs_tol=1e-12), f"{model_name} 类型覆盖率不符")
        check(math.isclose(float(row["token_coverage"]), occurrences / total_occurrences, rel_tol=0, abs_tol=1e-12), f"{model_name} token 覆盖率不符")

    vector_info = embedding_meta["pretrained_vectors"]
    vector_path = root / vector_info["path"]
    expected_known = sorted(train_vocabulary.intersection(ft_model.wv.key_to_index))
    check(vector_path.is_file(), "缺少监督训练词表预训练向量")
    exported_tokens: list[str] = []
    malformed_vectors = 0
    zero_vectors = 0
    with vector_path.open("r", encoding="utf-8") as handle:
        header = handle.readline().split()
        check(header == [str(len(expected_known)), "100"], ".vec 文件头与训练词表交集不符")
        for line in handle:
            parts = line.rstrip("\n").split(" ")
            if len(parts) != 101:
                malformed_vectors += 1
                continue
            exported_tokens.append(parts[0])
            try:
                values = [float(value) for value in parts[1:]]
            except ValueError:
                malformed_vectors += 1
                continue
            if not all(math.isfinite(value) for value in values):
                malformed_vectors += 1
            elif not any(value != 0 for value in values):
                zero_vectors += 1
    check(exported_tokens == expected_known, ".vec 包含范围不等于训练词表与 Gensim FastText 词表交集")
    check(malformed_vectors == 0 and zero_vectors == 0, ".vec 格式错误、含非有限值或零向量")
    check(vector_info["train_vocabulary_tokens"] == len(train_vocabulary), ".vec 训练词表规模不符")
    check(vector_info["exported_known_tokens"] == len(expected_known) and vector_info["oov_exported"] == 0, ".vec 导出范围包含 OOV 或数量错误")
    check(vector_path.stat().st_size == vector_info["bytes"], ".vec 文件大小与元数据不符")

    metric_rows = read_csv(results / "metrics.csv")
    metric_by_key = {(row["task"], row["method"], row["initialization"]): row for row in metric_rows}
    expected_keys = {
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
    check(len(metric_rows) == 15 and set(metric_by_key) == expected_keys, "指标文件缺少预设的 3 任务 × 5 方法组合")
    predictions = read_csv(results / "predictions.csv")
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in predictions:
        grouped[(row["task"], row["method"], row["initialization"], row["split"])].append(row)
    expected_groups = {(*key, split) for key in expected_keys for split in ("train", "val", "test")}
    check(set(grouped) == expected_groups, "逐篇预测没有覆盖每个方法的 train/val/test")
    for key in expected_keys:
        task, method, initialization = key
        for split in ("train", "val", "test"):
            rows = grouped[(*key, split)]
            labels = tasks[task]
            truth = [row["true_label"] for row in rows]
            guess = [row["predicted_label"] for row in rows]
            metric = metric_by_key[key]
            acc = float(accuracy_score(truth, guess))
            macro = float(f1_score(truth, guess, labels=labels, average="macro", zero_division=0))
            check(abs(float(metric[f"{split}_accuracy"]) - acc) < 1e-12, f"{task}/{method}/{initialization}/{split} Accuracy 汇总不符")
            check(abs(float(metric[f"{split}_macro_f1"]) - macro) < 1e-12, f"{task}/{method}/{initialization}/{split} Macro-F1 汇总不符")
            expected_n = sum(row["label"] in labels and row["split"] == split for row in selected)
            check(len(rows) == expected_n, f"{task}/{method}/{initialization}/{split} 预测行数不符")
    check(len(predictions) == 5 * sum(sum(row["label"] in tasks[task] for row in selected) for task in tasks), "逐篇预测总行数不符")

    curve_rows = read_csv(results / "fasttext_validation_curve.csv")
    check(len(curve_rows) == 18, "fastText 验证曲线没有完整覆盖 3 任务 × 2 初始化 × 3 epoch")
    curves_by_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in curve_rows:
        curves_by_key[(row["task"], row["initialization"])].append(row)
    for task in tasks:
        for initialization in ("random", "pretrained"):
            curves = curves_by_key[(task, initialization)]
            check({int(row["epoch"]) for row in curves} == {25, 100, 250}, f"{task}/{initialization} epoch 候选值缺失")
            picked = min(curves, key=lambda row: (-float(row["val_macro_f1"]), int(row["epoch"])))
            metric = metric_by_key[(task, "official_fasttext", initialization)]
            check(int(metric["selected_epoch"]) == int(picked["epoch"]), f"{task}/{initialization} 轮数不是按验证集 Macro-F1（平局取小轮数）选择")

    six_rows = grouped[("6class", "official_fasttext", "pretrained", "test")]
    matrix = confusion_matrix([row["true_label"] for row in six_rows], [row["predicted_label"] for row in six_rows], labels=categories)
    saved_matrix = read_csv(results / "confusion_matrix_6class_pretrained.csv")
    check(int(matrix.sum()) == split_target["test"] * len(categories), "六类测试混淆矩阵样本总数不符")
    check(len(saved_matrix) == len(categories), "六类混淆矩阵 CSV 行数不符")
    for i, row in enumerate(saved_matrix):
        check(row["true_label"] == categories[i], "混淆矩阵行顺序不符")
        for j, category in enumerate(categories):
            check(int(row[category]) == int(matrix[i, j]), f"混淆矩阵计数错误：{categories[i]}->{category}")
    errors = read_csv(results / "errors_6class_pretrained_fasttext.csv")
    actual_errors = [row for row in six_rows if row["true_label"] != row["predicted_label"]]
    check({row["article_id"] for row in errors} == {row["article_id"] for row in actual_errors}, "六类预训练错误审计表与逐篇预测不一致")
    for filename in ("data_split_counts.png", "classification_comparison.png", "confusion_6class_pretrained.png"):
        check((results / filename).is_file(), f"缺少图表文件：{filename}")

    report = {
        "status": "PASS",
        "teacher_corpus": {
            "source_archive_sha256": prep["source_lock"]["derived_source_archive_sha256"],
            "article_id_groups": prep["stats"]["article_id_groups"],
            "canonical_articles_after_integrity_and_dedup": len(prep["articles"]),
            "strict_gb18030_error_rows": len(prep["decode_errors"]),
            "strict_gb18030_error_articles": prep["stats"]["strict_gb18030_error_article_ids"],
            "same_id_conflicts": prep["stats"]["same_id_conflicting_segment_ids"],
            "same_id_duplicate_rows_dropped": prep["stats"]["same_id_same_body_duplicate_rows_dropped"],
            "canonical_duplicate_bodies_dropped": prep["stats"]["canonical_duplicate_body_articles_dropped"],
            "candidate_counts_by_label": prep["stats"]["candidate_counts_by_label"],
            "selected_per_class": target,
            "selected_articles": len(selected),
            "split_per_class": split_target,
            "train_validation_test_content_hashes_disjoint": True,
        },
        "nested_task_split_counts": nested_counts,
        "embedding_source": {
            "dataset": "bltlab/lr-sum",
            "revision": source_lock["revision"],
            "source_sha256": raw_hash,
            "source_rows": len(external_records),
            "included_articles": included_external,
            "included_tokens": token_total,
            "cross_corpus_exact_matches_any_teacher": all_teacher_exact,
            "cross_corpus_prefix_200_matches_any_teacher": all_teacher_prefix,
            "cross_corpus_exact_matches_validation_test": eval_exact,
            "cross_corpus_prefix_200_matches_validation_test": eval_prefix,
            "evaluation_text_absent_from_external_embeddings": True,
        },
        "embedding_checkpoints_loaded_and_parameters_verified": True,
        "embedding_checkpoint_reused": embedding_meta.get("checkpoint_reused_after_load_validation") is True,
        "train_vocabulary_types": len(train_vocabulary),
        "fasttext_known_train_vocabulary_types": len(expected_known),
        "pretrained_vec_rows": len(exported_tokens),
        "metric_groups_recomputed": len(expected_keys) * 3,
        "prediction_rows": len(predictions),
        "six_class_test_confusion_total": int(matrix.sum()),
        "six_class_test_errors": len(errors),
        "epoch_selection_uses_validation_only": True,
        "weak_label_cue_overlap": cue_overlap["by_split"],
    }
    (results / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    report = audit(project_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))

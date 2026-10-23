#!/usr/bin/env python3
"""Compare TF-IDF and fixed Gensim FastText means across label budgets."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

LABELS = ["体育", "经济", "文化", "国际", "政治", "社会"]
SEEDS = [17, 29, 43]
TRAIN_PER_CLASS = {0.15: 21, 0.40: 56, 1.00: 140}
TEST_COUNT = 180
CLASSIFIER = {"class": "sklearn.linear_model.LogisticRegression", "C": 1, "max_iter": 2000, "random_state": 42}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_and_validate_data() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows = read_csv(DATA / "frozen_split.csv")
    train_by_label: dict[str, list[dict[str, str]]] = {label: [] for label in LABELS}
    test_rows: list[dict[str, str]] = []
    for row in rows:
        if row["split"] == "train" and row["label"] in train_by_label:
            train_by_label[row["label"]].append(row)
        elif row["split"] == "test" and row["label"] in LABELS:
            test_rows.append(row)

    counts = {label: len(train_by_label[label]) for label in LABELS}
    if counts != {label: 140 for label in LABELS}:
        raise ValueError(f"Expected 140 fixed training articles per class; got {counts}")
    if len(test_rows) != TEST_COUNT or Counter(row["label"] for row in test_rows) != Counter({label: 30 for label in LABELS}):
        raise ValueError("Expected a fixed, balanced 180-article six-class test split")
    train_ids = {row["article_id"] for values in train_by_label.values() for row in values}
    test_ids = {row["article_id"] for row in test_rows}
    if train_ids & test_ids:
        raise ValueError("Frozen train and test article IDs overlap")
    return [row for label in LABELS for row in train_by_label[label]], sorted(test_rows, key=lambda row: row["article_id"])


def build_nested_samples(train_rows: list[dict[str, str]]) -> tuple[dict[tuple[int, float], list[dict[str, str]]], list[dict[str, Any]]]:
    import numpy as np

    by_label = {label: [row for row in train_rows if row["label"] == label] for label in LABELS}
    selections: dict[tuple[int, float], list[dict[str, str]]] = {}
    manifest: list[dict[str, Any]] = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        shuffled: dict[str, list[dict[str, str]]] = {}
        for label in LABELS:
            source_rows = by_label[label]
            permutation = rng.permutation(len(source_rows))
            shuffled[label] = [source_rows[int(index)] for index in permutation]

        for fraction, per_class in TRAIN_PER_CLASS.items():
            selected: list[dict[str, str]] = []
            for label in LABELS:
                selected.extend(shuffled[label][:per_class])
            # A stable article-ID order keeps classifier input independent of hash iteration order.
            selected.sort(key=lambda row: row["article_id"])
            if Counter(row["label"] for row in selected) != Counter({label: per_class for label in LABELS}):
                raise AssertionError("Training subset is not balanced")
            selections[(seed, fraction)] = selected
            run_name = f"seed_{seed}_fraction_{int(fraction * 100):03d}"
            rank_by_id = {
                row["article_id"]: rank
                for label in LABELS
                for rank, row in enumerate(shuffled[label][:per_class], start=1)
            }
            for row in selected:
                manifest.append({
                    "run": run_name,
                    "sampling_seed": seed,
                    "train_fraction": f"{fraction:.2f}",
                    "train_per_class": per_class,
                    "train_total": len(selected),
                    "selection_rank_within_class": rank_by_id[row["article_id"]],
                    "article_id": row["article_id"],
                    "label": row["label"],
                    "canonical_body_sha256": row["canonical_body_sha256"],
                })
    return selections, manifest


def mean_fasttext_vectors(rows: list[dict[str, str]], keyed_vectors: Any) -> tuple[Any, float, int, int]:
    import numpy as np

    started = time.perf_counter()
    values: list[Any] = []
    oov_types: set[str] = set()
    oov_tokens = 0
    for row in rows:
        vectors: list[Any] = []
        for token in row["body_text"].split():
            if token not in keyed_vectors.key_to_index:
                oov_types.add(token)
                oov_tokens += 1
            try:
                vectors.append(keyed_vectors.get_vector(token))
            except KeyError:
                continue
        if vectors:
            values.append(np.mean(np.asarray(vectors, dtype=np.float32), axis=0).astype(np.float32))
        else:
            values.append(np.zeros(keyed_vectors.vector_size, dtype=np.float32))
    return np.asarray(values, dtype=np.float32), time.perf_counter() - started, oov_tokens, len(oov_types)


def run_model(
    train_rows: list[dict[str, str]], test_rows: list[dict[str, str]], keyed_vectors: Any, method: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler

    started = time.perf_counter()
    train_texts = [row["body_text"] for row in train_rows]
    test_texts = [row["body_text"] for row in test_rows]
    y_train = [row["label"] for row in train_rows]
    y_test = [row["label"] for row in test_rows]

    if method == "tfidf_unigram":
        feature_started = time.perf_counter()
        vectorizer = TfidfVectorizer(
            tokenizer=str.split, preprocessor=None, token_pattern=None,
            lowercase=False, ngram_range=(1, 1), min_df=1, sublinear_tf=True,
        )
        x_train = vectorizer.fit_transform(train_texts)
        x_test = vectorizer.transform(test_texts)
        feature_seconds = time.perf_counter() - feature_started
        transform_details = {"feature_dimension": int(x_train.shape[1]), "scaler": "none"}
    elif method == "gensim_fasttext_mean":
        feature_started = time.perf_counter()
        x_train_raw, train_feature_seconds, train_oov_tokens, train_oov_types = mean_fasttext_vectors(train_rows, keyed_vectors)
        x_test_raw, test_feature_seconds, test_oov_tokens, test_oov_types = mean_fasttext_vectors(test_rows, keyed_vectors)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train_raw)
        x_test = scaler.transform(x_test_raw)
        feature_seconds = time.perf_counter() - feature_started
        transform_details = {
            "feature_dimension": int(x_train.shape[1]),
            "scaler": "StandardScaler fitted on selected training subset only",
            "train_fasttext_oov_tokens": train_oov_tokens,
            "train_fasttext_oov_types": train_oov_types,
            "test_fasttext_oov_tokens": test_oov_tokens,
            "test_fasttext_oov_types": test_oov_types,
            "train_mean_vector_seconds": train_feature_seconds,
            "test_mean_vector_seconds": test_feature_seconds,
        }
    else:
        raise ValueError(f"Unknown feature method: {method}")

    classifier = LogisticRegression(C=1, max_iter=2000, random_state=42)
    fit_started = time.perf_counter()
    classifier.fit(x_train, y_train)
    classifier_fit_seconds = time.perf_counter() - fit_started

    predict_started = time.perf_counter()
    predicted = classifier.predict(x_test).tolist()
    confidence = classifier.predict_proba(x_test).max(axis=1).astype(float).tolist()
    predict_seconds = time.perf_counter() - predict_started
    total_seconds = time.perf_counter() - started
    accuracy = float(accuracy_score(y_test, predicted))
    macro_f1 = float(f1_score(y_test, predicted, labels=LABELS, average="macro", zero_division=0))

    prediction_rows = [
        {
            "feature_method": method,
            "article_id": row["article_id"],
            "true_label": row["label"],
            "predicted_label": guess,
            "correct": str(guess == row["label"]).lower(),
            "confidence": f"{probability:.8f}",
        }
        for row, guess, probability in zip(test_rows, predicted, confidence)
    ]
    return {
        "test_accuracy": accuracy,
        "test_macro_f1": macro_f1,
        "feature_seconds": feature_seconds,
        "classifier_fit_seconds": classifier_fit_seconds,
        "prediction_seconds": predict_seconds,
        "pipeline_seconds": total_seconds,
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "feature_details": transform_details,
    }, prediction_rows


def make_figure(summary_rows: list[dict[str, Any]]) -> None:
    from matplotlib import font_manager
    import matplotlib.pyplot as plt

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    font_families = (
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
    )
    for family in font_families:
        if family in available_fonts:
            plt.rcParams["font.sans-serif"] = [family]
            break
    plt.rcParams["axes.unicode_minus"] = False

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    method_names = {
        "tfidf_unigram": "TF-IDF 一元词项",
        "gensim_fasttext_mean": "Gensim FastText 词向量均值",
    }
    method_colors = {"tfidf_unigram": "#176B87", "gensim_fasttext_mean": "#D97706"}
    fractions = [0.15, 0.40, 1.00]
    per_class = [TRAIN_PER_CLASS[fraction] for fraction in fractions]
    for axis, metric, title in (
        (axes[0], "macro_f1", "测试集 Macro-F1"),
        (axes[1], "accuracy", "测试集 Accuracy"),
    ):
        for method in method_names:
            rows = [row for row in summary_rows if row["feature_method"] == method]
            rows.sort(key=lambda row: float(row["train_fraction"]))
            means = [float(row[f"mean_test_{metric}"]) for row in rows]
            sds = [float(row[f"sd_test_{metric}"]) for row in rows]
            axis.errorbar(
                per_class, means, yerr=sds, marker="o", capsize=4, linewidth=2,
                label=method_names[method], color=method_colors[method],
            )
        axis.set_title(title)
        axis.set_xlabel("每类有标签训练文章数")
        axis.set_ylabel("得分")
        axis.set_xticks(per_class, [str(value) for value in per_class])
        axis.set_ylim(0.30, 1.00)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
    figure.suptitle("固定六类测试集上的标注规模曲线（测试集 180 篇）")
    path = RESULTS / "learning_curve_tfidf_vs_gensim_fasttext.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def run() -> dict[str, Any]:
    import gensim
    import numpy as np
    import sklearn
    from gensim.models import FastText

    split_path = DATA / "frozen_split.csv"
    corpus_stats = json.loads((DATA / "external_corpus_stats.json").read_text(encoding="utf-8"))
    embedding_meta = json.loads((RESULTS / "embedding_training.json").read_text(encoding="utf-8"))
    audit = json.loads((RESULTS / "audit_report.json").read_text(encoding="utf-8"))
    model_path = MODELS / "gensim_fasttext_sg.model"
    if not model_path.is_file():
        raise FileNotFoundError(f"Required reusable FastText checkpoint missing: {model_path}")
    if corpus_stats.get("source_labels_used") is not False:
        raise ValueError("The pinned embedding source metadata must confirm source_labels_used=false")
    if corpus_stats.get("cross_corpus_exact_matches_validation_test") != 0 or corpus_stats.get("cross_corpus_prefix_200_matches_validation_test") != 0:
        raise ValueError("Existing corpus-isolation audit found an exact or prefix match with validation/test articles")
    if embedding_meta.get("external_source_sha256") != corpus_stats.get("source_sha256"):
        raise ValueError("The saved FastText checkpoint metadata does not match the pinned external corpus source")
    if audit.get("status") != "PASS" or audit.get("teacher_corpus", {}).get("train_validation_test_content_hashes_disjoint") is not True:
        raise ValueError("Existing fixed-split audit is not PASS or reports split overlap")

    train_rows, test_rows = load_and_validate_data()
    selections, sample_manifest = build_nested_samples(train_rows)
    test_ids = [row["article_id"] for row in test_rows]
    test_id_sha256 = sha256_text("\n".join(test_ids))

    checkpoint_load_started = time.perf_counter()
    fasttext_model = FastText.load(str(model_path))
    checkpoint_load_seconds = time.perf_counter() - checkpoint_load_started
    keyed_vectors = fasttext_model.wv

    run_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    method_order = ["tfidf_unigram", "gensim_fasttext_mean"]
    for seed in SEEDS:
        for fraction, per_class in TRAIN_PER_CLASS.items():
            run_name = f"seed_{seed}_fraction_{int(fraction * 100):03d}"
            chosen = selections[(seed, fraction)]
            train_id_sha = sha256_text("\n".join(sorted(row["article_id"] for row in chosen)))
            for method in method_order:
                print(f"[run-start] {run_name} method={method} train={len(chosen)} test={len(test_rows)}", flush=True)
                result, predictions = run_model(chosen, test_rows, keyed_vectors, method)
                for prediction in predictions:
                    prediction.update({
                        "run": run_name,
                        "sampling_seed": seed,
                        "train_fraction": f"{fraction:.2f}",
                        "train_per_class": per_class,
                        "train_total": len(chosen),
                        "fixed_test_id_sha256": test_id_sha256,
                    })
                prediction_rows.extend(predictions)
                run_rows.append({
                    "run": run_name,
                    "sampling_seed": seed,
                    "train_fraction": f"{fraction:.2f}",
                    "train_per_class": per_class,
                    "train_total": len(chosen),
                    "test_total": len(test_rows),
                    "train_article_ids_sha256": train_id_sha,
                    "fixed_test_article_ids_sha256": test_id_sha256,
                    "feature_method": method,
                    "classifier": "LogisticRegression",
                    "classifier_C": 1,
                    "classifier_max_iter": 2000,
                    "test_accuracy": f"{result['test_accuracy']:.8f}",
                    "test_macro_f1": f"{result['test_macro_f1']:.8f}",
                    "feature_seconds": f"{result['feature_seconds']:.6f}",
                    "classifier_fit_seconds": f"{result['classifier_fit_seconds']:.6f}",
                    "prediction_seconds": f"{result['prediction_seconds']:.6f}",
                    "pipeline_seconds": f"{result['pipeline_seconds']:.6f}",
                    "feature_details_json": json.dumps(result["feature_details"], ensure_ascii=False, sort_keys=True),
                })
                print(
                    f"[run-done] {run_name} method={method} macro_f1={result['test_macro_f1']:.4f} "
                    f"accuracy={result['test_accuracy']:.4f} seconds={result['pipeline_seconds']:.2f}", flush=True,
                )

    summary_rows: list[dict[str, Any]] = []
    for method in method_order:
        for fraction, per_class in TRAIN_PER_CLASS.items():
            rows = [row for row in run_rows if row["feature_method"] == method and row["train_fraction"] == f"{fraction:.2f}"]
            accuracy_values = [float(row["test_accuracy"]) for row in rows]
            f1_values = [float(row["test_macro_f1"]) for row in rows]
            time_values = [float(row["pipeline_seconds"]) for row in rows]
            subset_hashes = {row["train_article_ids_sha256"] for row in rows}
            summary_rows.append({
                "feature_method": method,
                "train_fraction": f"{fraction:.2f}",
                "train_per_class": per_class,
                "train_total": per_class * len(LABELS),
                "test_total": len(test_rows),
                "sampling_seeds": len(rows),
                "unique_training_subsets": len(subset_hashes),
                "mean_test_accuracy": f"{statistics.mean(accuracy_values):.8f}",
                "sd_test_accuracy": f"{statistics.stdev(accuracy_values) if len(accuracy_values) > 1 else 0.0:.8f}",
                "mean_test_macro_f1": f"{statistics.mean(f1_values):.8f}",
                "sd_test_macro_f1": f"{statistics.stdev(f1_values) if len(f1_values) > 1 else 0.0:.8f}",
                "mean_pipeline_seconds": f"{statistics.mean(time_values):.6f}",
                "sd_pipeline_seconds": f"{statistics.stdev(time_values) if len(time_values) > 1 else 0.0:.6f}",
            })

    # Guard the analysis design and preserve a result row for every fixed test item.
    expected_runs = len(SEEDS) * len(TRAIN_PER_CLASS) * len(method_order)
    expected_predictions = expected_runs * len(test_rows)
    if len(run_rows) != expected_runs or len(prediction_rows) != expected_predictions:
        raise AssertionError(f"Expected {expected_runs} runs and {expected_predictions} predictions")
    if len({row["fixed_test_article_ids_sha256"] for row in run_rows}) != 1:
        raise AssertionError("Test split changed between runs")
    for row in prediction_rows:
        if row["article_id"] not in set(test_ids):
            raise AssertionError("Prediction contains an article outside the fixed test split")
    # Every smaller sample must be a prefix of the corresponding larger sample.
    for seed in SEEDS:
        small = {row["article_id"] for row in selections[(seed, 0.15)]}
        medium = {row["article_id"] for row in selections[(seed, 0.40)]}
        full = {row["article_id"] for row in selections[(seed, 1.00)]}
        if not small <= medium <= full:
            raise AssertionError(f"Nested-subset check failed for seed {seed}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    write_csv(RESULTS / "learning_curve_samples.csv", sample_manifest, [
        "run", "sampling_seed", "train_fraction", "train_per_class", "train_total",
        "selection_rank_within_class", "article_id", "label", "canonical_body_sha256",
    ])
    write_csv(RESULTS / "learning_curve_predictions.csv", prediction_rows, [
        "run", "sampling_seed", "train_fraction", "train_per_class", "train_total",
        "fixed_test_id_sha256", "feature_method", "article_id", "true_label", "predicted_label", "correct", "confidence",
    ])
    write_csv(RESULTS / "learning_curve_metrics.csv", run_rows, [
        "run", "sampling_seed", "train_fraction", "train_per_class", "train_total", "test_total",
        "train_article_ids_sha256", "fixed_test_article_ids_sha256", "feature_method", "classifier",
        "classifier_C", "classifier_max_iter", "test_accuracy", "test_macro_f1", "feature_seconds",
        "classifier_fit_seconds", "prediction_seconds", "pipeline_seconds", "feature_details_json",
    ])
    write_csv(RESULTS / "learning_curve_summary.csv", summary_rows, [
        "feature_method", "train_fraction", "train_per_class", "train_total", "test_total",
        "sampling_seeds", "unique_training_subsets", "mean_test_accuracy", "sd_test_accuracy",
        "mean_test_macro_f1", "sd_test_macro_f1", "mean_pipeline_seconds", "sd_pipeline_seconds",
    ])

    # Reload the persisted CSVs so the consistency check covers the actual row-level artifact.
    saved_prediction_rows = read_csv(RESULTS / "learning_curve_predictions.csv")
    saved_metric_rows = read_csv(RESULTS / "learning_curve_metrics.csv")
    if len(saved_prediction_rows) != expected_predictions or len(saved_metric_rows) != expected_runs:
        raise AssertionError("Saved prediction or metric CSV has an unexpected row count")

    # Recompute every run's metrics from the persisted per-article predictions.
    recomputed: dict[tuple[str, str], tuple[float, float]] = {}
    from sklearn.metrics import accuracy_score, f1_score
    prediction_groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in saved_prediction_rows:
        key = (row["run"], row["feature_method"])
        if row["correct"].lower() != str(row["true_label"] == row["predicted_label"]).lower():
            raise AssertionError(f"Saved correctness flag disagrees with labels for {key}")
        prediction_groups[key].append(row)
    metric_keys = {(row["run"], row["feature_method"]) for row in saved_metric_rows}
    if set(prediction_groups) != metric_keys or len(metric_keys) != expected_runs:
        raise AssertionError("Saved prediction groups do not match saved metric runs")

    for key, selected in prediction_groups.items():
        metric_row = next(row for row in saved_metric_rows if (row["run"], row["feature_method"]) == key)
        if len(selected) != int(metric_row["test_total"]):
            raise AssertionError(f"Unexpected saved prediction count for {key}")
        recomputed[key] = (
            float(accuracy_score([row["true_label"] for row in selected], [row["predicted_label"] for row in selected])),
            float(f1_score([row["true_label"] for row in selected], [row["predicted_label"] for row in selected], labels=LABELS, average="macro", zero_division=0)),
        )
    for row in saved_metric_rows:
        actual = recomputed[(row["run"], row["feature_method"])]
        if not math.isclose(actual[0], float(row["test_accuracy"]), abs_tol=1e-8) or not math.isclose(actual[1], float(row["test_macro_f1"]), abs_tol=1e-8):
            raise AssertionError("Metrics do not match the saved per-article predictions")
    make_figure(summary_rows)

    metadata = {
        "status": "PASS",
        "experiment": "six-class weak-label learning curve: TF-IDF unigram vs fixed Gensim FastText mean",
        "generated_at_local": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "environment": {
            "python": sys.version,
            "gensim": gensim.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "fasttext_checkpoint_load_seconds": checkpoint_load_seconds,
        },
        "data": {
            "fixed_split_file": "data/frozen_split.csv",
            "fixed_split_sha256": sha256_file(split_path),
            "train_articles_by_class": 140,
            "test_articles": len(test_rows),
            "test_articles_by_class": 30,
            "test_article_ids_sha256": test_id_sha256,
            "test_is_identical_in_all_runs": True,
            "validation_articles_not_used_for_training_or_selection": 180,
            "sampling_seeds": SEEDS,
            "fractions_and_train_articles_per_class": {f"{fraction:.2f}": count for fraction, count in TRAIN_PER_CLASS.items()},
            "nested_training_subsets_within_seed": True,
            "label_rule": "existing title-keyword weak labels; no manual labels added",
        },
        "pretrained_fasttext": {
            "checkpoint": "models/gensim_fasttext_sg.model",
            "checkpoint_sha256": sha256_file(model_path),
            "gensim_version": gensim.__version__,
            "embedding_source": corpus_stats["dataset"],
            "embedding_source_split": corpus_stats["split"],
            "embedding_source_revision": corpus_stats["revision"],
            "embedding_source_sha256": corpus_stats["source_sha256"],
            "embedding_source_labels_used": corpus_stats["source_labels_used"],
            "embedding_articles": corpus_stats["included_articles"],
            "embedding_tokens": corpus_stats["included_tokens"],
            "embedding_training_articles": embedding_meta["gensim_fasttext"]["corpus_count"],
            "embedding_training_seconds_from_checkpoint_metadata": embedding_meta["gensim_fasttext"]["seconds"],
            "teacher_exact_matches": corpus_stats["cross_corpus_exact_matches_any_teacher"],
            "teacher_prefix_200_matches": corpus_stats["cross_corpus_prefix_200_matches_any_teacher"],
            "test_exact_matches": corpus_stats["cross_corpus_exact_matches_validation_test"],
            "test_prefix_200_matches": corpus_stats["cross_corpus_prefix_200_matches_validation_test"],
            "word_vectors_retrained": False,
            "six_class_training_vocabulary_used_for_export_only": "The existing .vec export uses the union of all six-class training token types, but this experiment reads the saved Gensim checkpoint directly; it does not train on teacher-corpus text or labels.",
        },
        "models": {
            "tfidf_unigram": {
                "vectorizer": "TfidfVectorizer(tokenizer=str.split, lowercase=False, ngram_range=(1,1), sublinear_tf=True)",
                "classifier": CLASSIFIER,
            },
            "gensim_fasttext_mean": {
                "feature": "mean of fixed 100-dimensional FastTextKeyedVectors.get_vector(token) for each source token; OOV vectors use saved character n-gram buckets when available",
                "scaler": "StandardScaler fitted only on each selected training subset",
                "classifier": CLASSIFIER,
            },
        },
        "timing": "Per-run pipeline wall time includes feature construction, train-only scaling where applicable, classifier fitting, and fixed-test prediction; excludes checkpoint loading, output writing, and plot rendering. One-time checkpoint load is recorded separately.",
        "checks": {
            "existing_audit_status": audit["status"],
            "training_test_content_hashes_disjoint": audit["teacher_corpus"]["train_validation_test_content_hashes_disjoint"],
            "external_embedding_corpus_has_no_test_exact_or_prefix_match": corpus_stats["cross_corpus_exact_matches_validation_test"] == 0 and corpus_stats["cross_corpus_prefix_200_matches_validation_test"] == 0,
            "train_test_article_ids_disjoint": True,
            "class_balanced_nested_sampling": True,
            "same_test_split_all_runs": True,
            "saved_prediction_rows": len(saved_prediction_rows),
            "metrics_recomputed_from_saved_predictions": True,
        },
        "outputs": [
            "results/learning_curve_samples.csv",
            "results/learning_curve_predictions.csv",
            "results/learning_curve_metrics.csv",
            "results/learning_curve_summary.csv",
            "results/learning_curve_tfidf_vs_gensim_fasttext.png",
        ],
    }
    write_json(RESULTS / "learning_curve_metadata.json", metadata)
    print(f"[complete] status=PASS runs={len(run_rows)} predictions={len(prediction_rows)} figure=results/learning_curve_tfidf_vs_gensim_fasttext.png", flush=True)
    for row in summary_rows:
        print(
            f"[summary] {row['feature_method']} train={row['train_total']} "
            f"macro_f1={float(row['mean_test_macro_f1']):.4f}±{float(row['sd_test_macro_f1']):.4f} "
            f"accuracy={float(row['mean_test_accuracy']):.4f}±{float(row['sd_test_accuracy']):.4f} "
            f"seconds={float(row['mean_pipeline_seconds']):.2f}±{float(row['sd_pipeline_seconds']):.2f}", flush=True,
        )
    return metadata


if __name__ == "__main__":
    run()

#!/usr/bin/env python3
"""Freeze a small People Daily sample and compare embedding and fastText classifiers."""

from __future__ import annotations

import csv
from collections import Counter
import faulthandler
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from external_corpus import (
    EXPECTED_PARQUET_SHA256,
    RELATIVE_PARQUET,
    build_external_corpus,
    write_source_metadata,
)
from teacher_corpus import ARCHIVE_RELATIVE, build_teacher_corpus

# Keep every native numerical or model library within the four-thread task cap.
for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_var] = "1"


# Python only reads PYTHONHASHSEED when the process starts. Re-exec once so the
# documented `python run_experiment.py ...` entry point is deterministic too.
if __name__ == "__main__" and os.environ.get("PYTHONHASHSEED") != "0" and os.environ.get("GFT_HASH_REEXEC") != "1":
    child_env = os.environ.copy()
    child_env["PYTHONHASHSEED"] = "0"
    child_env["GFT_HASH_REEXEC"] = "1"
    raise SystemExit(subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env=child_env).returncode)


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
SCRATCH = Path(tempfile.gettempdir()) / f"pfr_fasttext_{os.getpid()}"

SEED = 42

CATEGORY_ORDER = ["体育", "经济", "文化", "国际", "政治", "社会"]
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "体育": ["体育", "足球", "篮球", "排球", "球队", "冠军", "世界杯", "奥运会", "奥运", "联赛", "运动员", "乒乓球", "网球", "赛场", "赛季", "球员", "赛事"],
    "经济": ["经济", "金融", "贸易", "市场", "企业", "公司", "银行", "投资", "股市", "股票", "工业", "农业", "财政", "经济学", "税收", "物价", "价格", "出口", "进口", "国企", "股价", "证券", "汇率"],
    "文化": ["文化", "文学", "艺术", "电影", "音乐", "作家", "演出", "戏剧", "美术", "电视", "诗歌", "小说", "出版", "书店", "舞蹈", "画家", "文艺", "图书", "舞剧", "歌剧", "话剧", "舞台"],
    "国际": ["美国", "日本", "俄罗斯", "英国", "法国", "德国", "联合国", "朝鲜", "韩国", "伊拉克", "伊朗", "南非", "外交", "加拿大", "澳大利亚", "以色列", "巴勒斯坦", "新加坡", "越南", "东盟", "印度", "巴基斯坦", "尼加拉瓜", "墨西哥", "意大利", "西班牙", "荷兰", "瑞士", "瑞典", "乌克兰", "菲律宾", "印尼", "柬埔寨", "缅甸", "泰国", "马来西亚", "阿富汗"],
    "政治": ["政府", "党", "人大", "政协", "国务院", "总书记", "代表大会", "政治", "党中央", "中共中央", "党代会", "民主", "法制", "党政", "议会", "选举", "国会"],
    "社会": ["医疗", "医院", "卫生", "民生", "就业", "农民", "城市", "交通", "灾害", "消费", "生活", "环境", "住房", "事故", "工人", "职工", "居民", "社会保障", "教育", "学校", "学生", "治安", "犯罪", "妇女", "儿童", "养老", "食品安全"],
}
TASKS: dict[str, list[str]] = {
    "2class": ["体育", "经济"],
    "4class": ["体育", "经济", "文化", "国际"],
    "6class": CATEGORY_ORDER.copy(),
}
def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def salted_hash(salt: str, article_id: str) -> str:
    return sha256_text(salt + article_id)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def classify_title(title_tokens: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    matched: dict[str, list[str]] = {}
    for category in CATEGORY_ORDER:
        found = sorted({token for token in title_tokens if token in CATEGORY_KEYWORDS[category]})
        if found:
            matched[category] = found
    return list(matched), matched


def maybe_save_gensim_model(model: Any, path: Path) -> dict[str, Any]:
    model.save(str(path))
    size = path.stat().st_size
    return {"path": str(path.relative_to(ROOT)), "bytes": size, "under_100_mib": size < 100 * 1024 * 1024}


def write_train_vocabulary_artifacts(w2v: Any, ft: Any) -> tuple[list[str], list[dict[str, Any]], Path]:
    selected = read_csv(DATA / "frozen_split.csv")
    train_token_counts = Counter(
        token for row in selected if row["split"] == "train" for token in row["body_text"].split()
    )
    train_vocab = sorted(train_token_counts)
    coverage_rows: list[dict[str, Any]] = []
    for name, keyed_vectors in (("word2vec", w2v.wv), ("gensim_fasttext", ft.wv)):
        known = set(train_vocab).intersection(keyed_vectors.key_to_index)
        known_occurrences = sum(train_token_counts[token] for token in known)
        total_occurrences = sum(train_token_counts.values())
        coverage_rows.append({
            "model": name,
            "train_vocabulary_types": len(train_vocab),
            "known_types": len(known),
            "type_coverage": len(known) / len(train_vocab) if train_vocab else 0,
            "train_token_occurrences": total_occurrences,
            "known_token_occurrences": known_occurrences,
            "token_coverage": known_occurrences / total_occurrences if total_occurrences else 0,
        })
    write_csv(
        RESULTS / "train_vocabulary_coverage.csv",
        coverage_rows,
        ["model", "train_vocabulary_types", "known_types", "type_coverage", "train_token_occurrences", "known_token_occurrences", "token_coverage"],
    )
    pretrained_path = MODELS / "gensim_fasttext_train_vocab.vec"
    written = 0
    with pretrained_path.open("w", encoding="utf-8", newline="\n") as handle:
        known = [token for token in train_vocab if token in ft.wv.key_to_index]
        handle.write(f"{len(known)} {ft.vector_size}\n")
        for token in known:
            vector = ft.wv.get_vector(token)
            handle.write(token + " " + " ".join(f"{float(value):.8f}" for value in vector) + "\n")
            written += 1
    return train_vocab, coverage_rows, pretrained_path


def train_embeddings() -> tuple[Any, Any, dict[str, Any]]:
    from gensim.models import FastText, Word2Vec
    from gensim.models.word2vec import LineSentence

    corpus_path = DATA / "unsupervised_corpus.txt"
    if not corpus_path.is_file() or corpus_path.stat().st_size == 0:
        raise FileNotFoundError(f"缺少未标注词向量语料: {corpus_path}")
    MODELS.mkdir(parents=True, exist_ok=True)
    saved_meta_path = RESULTS / "embedding_training.json"
    w2v_path = MODELS / "word2vec_sg.model"
    ft_path = MODELS / "gensim_fasttext_sg.model"
    corpus_hash = sha256_bytes(corpus_path.read_bytes())
    external_stats = json.loads((DATA / "external_corpus_stats.json").read_text(encoding="utf-8"))
    if saved_meta_path.is_file() and w2v_path.is_file() and ft_path.is_file():
        cached_meta = json.loads(saved_meta_path.read_text(encoding="utf-8"))
        if (
            cached_meta.get("unsupervised_corpus_sha256") == corpus_hash
            and cached_meta.get("external_source_sha256") == external_stats["source_sha256"]
            and cached_meta.get("external_revision") == external_stats["revision"]
        ):
            print("[stage] reuse hash-matched completed embedding checkpoint", flush=True)
            w2v = Word2Vec.load(str(w2v_path))
            ft = FastText.load(str(ft_path))
            assert w2v.corpus_count == ft.corpus_count == 482
            assert all(
                model.sg == 1 and model.vector_size == 100 and model.window == 5
                and model.min_count == 3 and model.epochs == 10 and model.workers == 1
                and model.seed == SEED
                for model in (w2v, ft)
            )
            train_vocab, coverage_rows, pretrained_path = write_train_vocabulary_artifacts(w2v, ft)
            expected_train_vocab = len(train_vocab)
            known_ft = sum(token in ft.wv.key_to_index for token in train_vocab)
            cached_meta["pretrained_vectors"] = {
                "path": str(pretrained_path.relative_to(ROOT)),
                "bytes": pretrained_path.stat().st_size,
                "train_vocabulary_tokens": expected_train_vocab,
                "exported_known_tokens": known_ft,
                "oov_exported": 0,
            }
            cached_meta["train_vocabulary_coverage"] = coverage_rows
            cached_meta["checkpoint_reused_after_load_validation"] = True
            cached_meta["thread_environment"] = "not_recorded_for_embedding_checkpoint"
            write_json(saved_meta_path, cached_meta)
            return w2v, ft, cached_meta
    print(f"[stage] Word2Vec SG training start corpus_sha256={corpus_hash}", flush=True)
    sentence_source = LineSentence(str(corpus_path))
    start = time.perf_counter()
    w2v = Word2Vec(
        vector_size=100, window=5, min_count=3, workers=1, seed=SEED,
        sg=1, epochs=10, sorted_vocab=1,
    )
    w2v.build_vocab(sentence_source)
    w2v.train(sentence_source, total_examples=w2v.corpus_count, epochs=10)
    w2v_seconds = time.perf_counter() - start
    w2v_meta = maybe_save_gensim_model(w2v, MODELS / "word2vec_sg.model")
    print(f"[stage] Word2Vec SG done seconds={w2v_seconds:.2f} vocab={len(w2v.wv)}", flush=True)

    print("[stage] Gensim FastText SG training start", flush=True)
    sentence_source = LineSentence(str(corpus_path))
    start = time.perf_counter()
    ft = FastText(
        vector_size=100, window=5, min_count=3, workers=1, seed=SEED,
        sg=1, epochs=10, min_n=2, max_n=4, bucket=100000, sorted_vocab=1,
    )
    ft.build_vocab(sentence_source)
    ft.train(sentence_source, total_examples=ft.corpus_count, epochs=10)
    ft_seconds = time.perf_counter() - start
    ft_meta = maybe_save_gensim_model(ft, MODELS / "gensim_fasttext_sg.model")
    print(f"[stage] Gensim FastText SG done seconds={ft_seconds:.2f} vocab={len(ft.wv)}", flush=True)

    train_vocab, coverage_rows, pretrained_path = write_train_vocabulary_artifacts(w2v, ft)
    written = sum(token in ft.wv.key_to_index for token in train_vocab)

    meta = {
        "w2v": {"seconds": w2v_seconds, "vocabulary_size": len(w2v.wv), "corpus_count": w2v.corpus_count, "artifact": w2v_meta, "params": {"sg": 1, "vector_size": 100, "window": 5, "min_count": 3, "epochs": 10, "workers": 1, "seed": SEED}},
        "gensim_fasttext": {"seconds": ft_seconds, "vocabulary_size": len(ft.wv), "corpus_count": ft.corpus_count, "artifact": ft_meta, "params": {"sg": 1, "vector_size": 100, "window": 5, "min_count": 3, "epochs": 10, "workers": 1, "seed": SEED, "min_n": 2, "max_n": 4, "bucket": 100000}},
        "pretrained_vectors": {"path": str(pretrained_path.relative_to(ROOT)), "bytes": pretrained_path.stat().st_size, "train_vocabulary_tokens": len(train_vocab), "exported_known_tokens": written, "oov_exported": 0},
        "train_vocabulary_coverage": coverage_rows,
        "unsupervised_corpus_sha256": sha256_bytes(corpus_path.read_bytes()),
        "external_source_sha256": json.loads((DATA / "external_corpus_stats.json").read_text(encoding="utf-8"))["source_sha256"],
        "external_revision": json.loads((DATA / "external_corpus_stats.json").read_text(encoding="utf-8"))["revision"],
    }
    write_json(saved_meta_path, meta)
    return w2v, ft, meta


def embedding_diagnostics(w2v: Any, ft: Any) -> None:
    import numpy as np

    rows: list[dict[str, Any]] = []
    terms = ["经济", "银行", "足球", "文化"]
    oov_terms = ["数字农业实验室", "星际体育学院", "云端金融平台"]
    for model_name, model in (("gensim_word2vec", w2v.wv), ("gensim_fasttext", ft.wv)):
        for term in terms:
            in_vocab = term in model.key_to_index
            vector = model.get_vector(term) if in_vocab or model_name == "gensim_fasttext" else None
            neighbors = model.most_similar(positive=[term], topn=5) if vector is not None else []
            rows.append({
                "model": model_name,
                "query": term,
                "key_to_index": in_vocab,
                "vector_available": vector is not None,
                "vector_norm": float(np.linalg.norm(vector)) if vector is not None else "",
                "neighbors": " | ".join(f"{word}:{score:.4f}" for word, score in neighbors),
                "interpretation": "diagnostic_only_not_a_quality_metric",
            })
    for term in oov_terms:
        in_vocab = term in ft.wv.key_to_index
        try:
            vector = ft.wv.get_vector(term)
            neighbors = ft.wv.most_similar(positive=[term], topn=5)
            rows.append({
                "model": "gensim_fasttext",
                "query": term,
                "key_to_index": in_vocab,
                "vector_available": True,
                "vector_norm": float(np.linalg.norm(vector)),
                "neighbors": " | ".join(f"{word}:{score:.4f}" for word, score in neighbors),
                "interpretation": "diagnostic_only_subword_composition_not_semantic_validation",
            })
        except KeyError as exc:
            rows.append({"model": "gensim_fasttext", "query": term, "key_to_index": in_vocab, "vector_available": False, "vector_norm": "", "neighbors": str(exc), "interpretation": "diagnostic_only_not_a_quality_metric"})
    write_csv(RESULTS / "embedding_diagnostics.csv", rows, ["model", "query", "key_to_index", "vector_available", "vector_norm", "neighbors", "interpretation"])


def load_samples() -> list[dict[str, str]]:
    rows = read_csv(DATA / "frozen_split.csv")
    if not rows:
        raise ValueError("冻结样本为空")
    return rows


def metric_pair(y_true: list[str], y_pred: list[str], labels: list[str]) -> tuple[float, float]:
    from sklearn.metrics import accuracy_score, f1_score

    return float(accuracy_score(y_true, y_pred)), float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def append_predictions(
    target: list[dict[str, Any]], task: str, method: str, initialization: str,
    split: str, rows: list[dict[str, str]], predicted: list[str], confidence: list[float],
) -> tuple[float, float]:
    labels = TASKS[task]
    truths = [row["label"] for row in rows]
    accuracy, macro_f1 = metric_pair(truths, predicted, labels)
    for row, guess, conf in zip(rows, predicted, confidence):
        target.append({
            "task": task,
            "method": method,
            "initialization": initialization,
            "split": split,
            "article_id": row["article_id"],
            "true_label": row["label"],
            "predicted_label": guess,
            "correct": guess == row["label"],
            "confidence": f"{conf:.8f}",
        })
    return accuracy, macro_f1


def prediction_metrics(pred_rows: list[dict[str, Any]], task: str, method: str, init: str, split: str) -> tuple[float, float]:
    rows = [row for row in pred_rows if row["task"] == task and row["method"] == method and row["initialization"] == init and row["split"] == split]
    return metric_pair([str(row["true_label"]) for row in rows], [str(row["predicted_label"]) for row in rows], TASKS[task])


def fit_sklearn_baseline(
    rows: list[dict[str, str]], task: str, method: str, wv: Any | None, pretrain_seconds: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    labels = TASKS[task]
    train = [row for row in rows if row["split"] == "train" and row["label"] in labels]
    val = [row for row in rows if row["split"] == "val" and row["label"] in labels]
    test = [row for row in rows if row["split"] == "test" and row["label"] in labels]
    start = time.perf_counter()
    if method == "tfidf_logreg":
        vectorizer = TfidfVectorizer(
            tokenizer=str.split, preprocessor=None, token_pattern=None,
            lowercase=False, ngram_range=(1, 1), min_df=1, sublinear_tf=True,
        )
        x_train = vectorizer.fit_transform([row["body_text"] for row in train])
        x_val = vectorizer.transform([row["body_text"] for row in val])
        x_test = vectorizer.transform([row["body_text"] for row in test])
        clf = LogisticRegression(C=1, max_iter=2000, random_state=SEED)
        clf.fit(x_train, [row["label"] for row in train])
        predict = lambda x: (clf.predict(x).tolist(), clf.predict_proba(x).max(axis=1).astype(float).tolist())
    else:
        assert wv is not None
        def to_vector(text: str) -> list[float]:
            pieces: list[Any] = []
            for token in text.split():
                if token in wv.key_to_index:
                    pieces.append(wv.get_vector(token))
                elif method == "fasttext_mean_logreg":
                    try:
                        pieces.append(wv.get_vector(token))
                    except KeyError:
                        continue
            if pieces:
                return np.mean(np.asarray(pieces, dtype=np.float32), axis=0).astype(np.float32).tolist()
            return np.zeros(wv.vector_size, dtype=np.float32).tolist()
        raw_train = np.asarray([to_vector(row["body_text"]) for row in train], dtype=np.float32)
        raw_val = np.asarray([to_vector(row["body_text"]) for row in val], dtype=np.float32)
        raw_test = np.asarray([to_vector(row["body_text"]) for row in test], dtype=np.float32)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(raw_train)
        x_val = scaler.transform(raw_val)
        x_test = scaler.transform(raw_test)
        clf = LogisticRegression(C=1, max_iter=2000, random_state=SEED)
        clf.fit(x_train, [row["label"] for row in train])
        predict = lambda x: (clf.predict(x).tolist(), clf.predict_proba(x).max(axis=1).astype(float).tolist())
    fit_seconds = time.perf_counter() - start
    predictions: list[dict[str, Any]] = []
    split_metrics: dict[str, tuple[float, float]] = {}
    for split, split_rows, values in (("train", train, x_train), ("val", val, x_val), ("test", test, x_test)):
        guesses, confidence = predict(values)
        split_metrics[split] = append_predictions(predictions, task, method, "", split, split_rows, guesses, confidence)
    record = {
        "task": task, "method": method, "initialization": "", "selected_epoch": "",
        "train_n": len(train), "val_n": len(val), "test_n": len(test),
        "train_accuracy": split_metrics["train"][0], "train_macro_f1": split_metrics["train"][1],
        "val_accuracy": split_metrics["val"][0], "val_macro_f1": split_metrics["val"][1],
        "test_accuracy": split_metrics["test"][0], "test_macro_f1": split_metrics["test"][1],
        "fit_seconds": fit_seconds, "representation_pretrain_seconds": pretrain_seconds,
        "parameters_json": json.dumps({"classifier": "LogisticRegression", "C": 1, "max_iter": 2000, "random_state": SEED, "features": "unigram TF-IDF" if method == "tfidf_logreg" else "frozen mean embedding + train-only StandardScaler"}, ensure_ascii=False, sort_keys=True),
    }
    return record, predictions


def train_fasttext_candidates(rows: list[dict[str, str]], task: str, initialization: str, pretrained_path: Path | None) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], Any]:
    import fasttext
    import shutil

    labels = TASKS[task]
    train = [row for row in rows if row["split"] == "train" and row["label"] in labels]
    val = [row for row in rows if row["split"] == "val" and row["label"] in labels]
    test = [row for row in rows if row["split"] == "test" and row["label"] in labels]
    SCRATCH.mkdir(parents=True, exist_ok=True)
    input_path = SCRATCH / f"fasttext_{task}_{initialization}_train.txt"
    with input_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in train:
            handle.write(f"__label__{row['label']} {row['body_text']}\n")

    validation_curve: list[dict[str, Any]] = []
    best_model = None
    best_key: tuple[float, int] | None = None
    best_epoch_seconds = 0.0
    for epoch in (25, 100, 250):
        print(f"[fasttext-start] task={task} init={initialization} epoch={epoch}", flush=True)
        kwargs: dict[str, Any] = {
            "input": str(input_path), "lr": 0.5, "epoch": epoch, "wordNgrams": 2,
            "minn": 0, "maxn": 0, "loss": "softmax", "dim": 100,
            "thread": 1, "bucket": 100000, "seed": SEED, "verbose": 0,
        }
        if initialization == "pretrained":
            if pretrained_path is None or not pretrained_path.is_file():
                raise FileNotFoundError("缺少 Gensim FastText 导出的监督训练词表 .vec")
            # fastText's Windows C++ loader also uses narrow paths for .vec files.
            scratch_vectors = SCRATCH / "gensim_fasttext_train_vocab.vec"
            if not scratch_vectors.is_file() or scratch_vectors.stat().st_size != pretrained_path.stat().st_size:
                shutil.copyfile(pretrained_path, scratch_vectors)
            kwargs["pretrainedVectors"] = str(scratch_vectors)
        start = time.perf_counter()
        model = fasttext.train_supervised(**kwargs)
        seconds = time.perf_counter() - start
        guesses: list[str] = []
        for row in val:
            predicted_labels, _ = model.predict(row["body_text"], k=1)
            guesses.append(predicted_labels[0].removeprefix("__label__") if predicted_labels else "")
        accuracy, macro_f1 = metric_pair([row["label"] for row in val], guesses, labels)
        print(f"[fasttext-done] task={task} init={initialization} epoch={epoch} val_macro_f1={macro_f1:.4f} seconds={seconds:.2f}", flush=True)
        validation_curve.append({
            "task": task, "initialization": initialization, "epoch": epoch,
            "val_n": len(val), "val_accuracy": accuracy, "val_macro_f1": macro_f1,
            "fit_seconds": seconds,
        })
        candidate_key = (-macro_f1, epoch)
        if best_key is None or candidate_key < best_key:
            best_key = candidate_key
            best_model = model
            best_epoch_seconds = seconds
    assert best_model is not None and best_key is not None

    predictions: list[dict[str, Any]] = []
    split_metrics: dict[str, tuple[float, float]] = {}
    for split, split_rows in (("train", train), ("val", val), ("test", test)):
        guesses = []
        confidences = []
        for row in split_rows:
            predicted_labels, probabilities = best_model.predict(row["body_text"], k=1)
            guesses.append(predicted_labels[0].removeprefix("__label__") if predicted_labels else "")
            confidences.append(float(probabilities[0]) if probabilities else 0.0)
        split_metrics[split] = append_predictions(predictions, task, "official_fasttext", initialization, split, split_rows, guesses, confidences)
    selected_epoch = best_key[1]
    record = {
        "task": task, "method": "official_fasttext", "initialization": initialization,
        "selected_epoch": selected_epoch, "train_n": len(train), "val_n": len(val), "test_n": len(test),
        "train_accuracy": split_metrics["train"][0], "train_macro_f1": split_metrics["train"][1],
        "val_accuracy": split_metrics["val"][0], "val_macro_f1": split_metrics["val"][1],
        "test_accuracy": split_metrics["test"][0], "test_macro_f1": split_metrics["test"][1],
        "fit_seconds": best_epoch_seconds, "representation_pretrain_seconds": 0.0,
        "parameters_json": json.dumps({"implementation": "official fastText Python binding fasttext-wheel", "loss": "softmax", "dim": 100, "lr": 0.5, "selected_epoch": selected_epoch, "candidate_epochs": [25, 100, 250], "wordNgrams": 2, "minn": 0, "maxn": 0, "thread": 1, "bucket": 100000, "seed": SEED, "initialization": initialization, "pretrainedVectors": str(pretrained_path.relative_to(ROOT)) if initialization == "pretrained" and pretrained_path else "random"}, ensure_ascii=False, sort_keys=True),
    }
    if task == "6class" and initialization == "pretrained":
        MODELS.mkdir(parents=True, exist_ok=True)
        saved = MODELS / "official_fasttext_pretrained_6class.bin"
        scratch_model = SCRATCH / "official_fasttext_pretrained_6class.bin"
        best_model.save_model(str(scratch_model))
        shutil.copyfile(scratch_model, saved)
        size = saved.stat().st_size
        if size >= 100 * 1024 * 1024:
            saved.unlink()
            record["model_saved"] = "skipped: artifact exceeded 100 MiB"
        else:
            record["model_saved"] = str(saved.relative_to(ROOT))
            record["model_bytes"] = size
    return record, predictions, validation_curve, best_model


def font_setup() -> None:
    import matplotlib
    from matplotlib import font_manager

    candidates = [Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\simhei.ttf"), Path(r"C:\Windows\Fonts\simsun.ttc")]
    font_path = next((path for path in candidates if path.is_file()), None)
    if font_path:
        font_manager.fontManager.addfont(str(font_path))
        family = font_manager.FontProperties(fname=str(font_path)).get_name()
        matplotlib.rcParams["font.sans-serif"] = [family]
    matplotlib.rcParams["axes.unicode_minus"] = False


def make_charts(metrics_rows: list[dict[str, Any]], predictions: list[dict[str, Any]], curves: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import confusion_matrix

    font_setup()
    RESULTS.mkdir(parents=True, exist_ok=True)
    tasks = ["2class", "4class", "6class"]
    method_order = ["tfidf_logreg", "w2v_mean_logreg", "fasttext_mean_logreg", "official_fasttext_random", "official_fasttext_pretrained"]
    method_names = {"tfidf_logreg": "TF-IDF + LR", "w2v_mean_logreg": "Word2Vec mean + LR", "fasttext_mean_logreg": "Gensim FastText mean + LR", "official_fasttext_random": "Official fastText random", "official_fasttext_pretrained": "Official fastText pretrained"}

    figure, axes = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(len(tasks))
    width = 0.15
    for method_index, method in enumerate(method_order):
        values = []
        accs = []
        for task in tasks:
            init = "pretrained" if method == "official_fasttext_pretrained" else "random" if method == "official_fasttext_random" else ""
            row = next(row for row in metrics_rows if row["task"] == task and row["method"] == ("official_fasttext" if init else method) and row["initialization"] == init)
            values.append(float(row["test_macro_f1"]))
            accs.append(float(row["test_accuracy"]))
        axes[0].bar(x + (method_index - 2) * width, values, width, label=method_names[method])
        axes[1].bar(x + (method_index - 2) * width, accs, width, label=method_names[method])
    for axis, title in zip(axes, ["测试集 Macro-F1", "测试集 Accuracy"]):
        axis.set_title(title)
        axis.set_xticks(x, ["2 类", "4 类", "6 类"])
        axis.set_ylim(0, 1.05)
        axis.set_ylabel("Score")
        axis.grid(axis="y", alpha=0.2)
    handles, legend_labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, legend_labels, frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=3)
    figure.suptitle("Gensim 表示与官方 fastText 分类对比")
    figure.tight_layout(rect=(0, 0.12, 1, 0.94))
    figure.savefig(RESULTS / "classification_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    split_rows = read_csv(DATA / "frozen_split.csv")
    class_names = CATEGORY_ORDER
    split_values = {
        split: [sum(row["label"] == label and row["split"] == split for row in split_rows) for label in class_names]
        for split in ("train", "val", "test")
    }
    corpus_stats = json.loads((DATA / "corpus_stats.json").read_text(encoding="utf-8"))
    candidate_total = sum(corpus_stats["candidate_counts_by_label"].values())
    flow_labels = ["文章 ID 分组", "有效且正文去重", "单类关键词候选", "固定分类样本"]
    flow_values = [
        corpus_stats["article_id_groups"],
        corpus_stats["canonical_articles_after_integrity_and_dedup"],
        candidate_total,
        corpus_stats["selected_articles"],
    ]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    axes[0].barh(flow_labels, flow_values, color=["#667085", "#176B87", "#D97706", "#57A773"])
    axes[0].invert_yaxis()
    axes[0].set_xscale("log")
    axes[0].set_xlabel("文章数（对数刻度）")
    axes[0].set_title("教师语料筛选规模")
    for index, value in enumerate(flow_values):
        axes[0].text(value * 1.04, index, f"{value:,}", va="center")
    bottom = np.zeros(len(class_names))
    colors = {"train": "#176B87", "val": "#D97706", "test": "#57A773"}
    for split in ("train", "val", "test"):
        axes[1].bar(class_names, split_values[split], bottom=bottom, label={"train": "训练", "val": "验证", "test": "测试"}[split], color=colors[split])
        bottom += np.asarray(split_values[split])
    axes[1].set_ylabel("文章数")
    axes[1].set_title("六类固定样本与拆分")
    axes[1].legend(frameon=False)
    figure.suptitle("教师新闻语料分类样本与拆分计数")
    figure.savefig(RESULTS / "data_split_counts.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    six = [row for row in metrics_rows if row["task"] == "6class" and row["method"] == "official_fasttext" and row["initialization"] == "pretrained"][0]
    test_rows = [row for row in predictions if row["task"] == "6class" and row["method"] == "official_fasttext" and row["initialization"] == "pretrained" and row["split"] == "test"]
    y_true = [row["true_label"] for row in test_rows]
    y_pred = [row["predicted_label"] for row in test_rows]
    cm = confusion_matrix(y_true, y_pred, labels=CATEGORY_ORDER)
    write_csv(RESULTS / "confusion_matrix_6class_pretrained.csv", [{"true_label": label, **{pred: int(cm[i, j]) for j, pred in enumerate(CATEGORY_ORDER)}} for i, label in enumerate(CATEGORY_ORDER)], ["true_label", *CATEGORY_ORDER])
    figure, axis = plt.subplots(figsize=(8.0, 7.0), constrained_layout=True)
    image = axis.imshow(cm, cmap="Blues")
    axis.set_xticks(np.arange(len(CATEGORY_ORDER)), CATEGORY_ORDER, rotation=35, ha="right")
    axis.set_yticks(np.arange(len(CATEGORY_ORDER)), CATEGORY_ORDER)
    axis.set_xlabel("预测类别")
    axis.set_ylabel("真实类别")
    axis.set_title(f"六类官方 fastText（预训练初始化，epoch={six['selected_epoch']}）")
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axis.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > threshold else "black")
    figure.colorbar(image, ax=axis, shrink=0.8)
    figure.savefig(RESULTS / "confusion_6class_pretrained.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

def run_training() -> dict[str, Any]:
    import fasttext
    import gensim
    import importlib.metadata
    import matplotlib
    import numpy
    import pyarrow
    import scipy
    import sklearn
    import jieba

    print("[stage] parse six-month GB18030 teacher source and freeze sample", flush=True)
    teacher_started = time.perf_counter()
    teacher_data = build_teacher_corpus(ROOT, write_outputs=True)
    teacher_elapsed = time.perf_counter() - teacher_started
    print(
        f"[stage] teacher corpus ready articles={len(teacher_data['articles'])} "
        f"candidates={len(teacher_data['candidate_articles'])} selected={len(teacher_data['selected_rows'])} "
        f"seconds={teacher_elapsed:.2f}",
        flush=True,
    )
    print("[stage] normalize and tokenize pinned LR-Sum corpus", flush=True)
    external_started = time.perf_counter()
    external_stats = build_external_corpus(ROOT, teacher_articles=teacher_data["articles"])
    external_stats["preprocessing_seconds"] = time.perf_counter() - external_started
    write_source_metadata(ROOT, teacher_data["source_lock"])
    print(f"[stage] LR-Sum corpus ready articles={external_stats['included_articles']} tokens={external_stats['included_tokens']} seconds={external_stats['preprocessing_seconds']:.2f}", flush=True)

    rows = load_samples()
    RESULTS.mkdir(parents=True, exist_ok=True)
    print("[stage] train embeddings", flush=True)
    embedding_started = time.perf_counter()
    w2v, ft, embedding_meta = train_embeddings()
    embedding_elapsed = time.perf_counter() - embedding_started
    print(f"[stage] embeddings ready seconds={embedding_elapsed:.2f}", flush=True)
    embedding_diagnostics(w2v, ft)
    pretrained_path = MODELS / "gensim_fasttext_train_vocab.vec"
    metrics_rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    validation_curves: list[dict[str, Any]] = []
    classification_started = time.perf_counter()
    for task in TASKS:
        print(f"[task-start] {task}", flush=True)
        row, preds = fit_sklearn_baseline(rows, task, "tfidf_logreg", None, 0.0)
        metrics_rows.append(row)
        predictions.extend(preds)
        row, preds = fit_sklearn_baseline(rows, task, "w2v_mean_logreg", w2v.wv, embedding_meta["w2v"]["seconds"])
        metrics_rows.append(row)
        predictions.extend(preds)
        row, preds = fit_sklearn_baseline(rows, task, "fasttext_mean_logreg", ft.wv, embedding_meta["gensim_fasttext"]["seconds"])
        metrics_rows.append(row)
        predictions.extend(preds)
        for initialization, vec_path in (("random", None), ("pretrained", pretrained_path)):
            row, preds, curve, _model = train_fasttext_candidates(rows, task, initialization, vec_path)
            metrics_rows.append(row)
            predictions.extend(preds)
            validation_curves.extend(curve)

    classification_elapsed = time.perf_counter() - classification_started
    print(f"[stage] classification candidates done seconds={classification_elapsed:.2f}", flush=True)
    metric_fields = ["task", "method", "initialization", "selected_epoch", "train_n", "val_n", "test_n", "train_accuracy", "train_macro_f1", "val_accuracy", "val_macro_f1", "test_accuracy", "test_macro_f1", "fit_seconds", "representation_pretrain_seconds", "parameters_json", "model_saved", "model_bytes"]
    write_csv(RESULTS / "metrics.csv", metrics_rows, metric_fields)
    write_csv(RESULTS / "predictions.csv", predictions, ["task", "method", "initialization", "split", "article_id", "true_label", "predicted_label", "correct", "confidence"])
    write_csv(RESULTS / "fasttext_validation_curve.csv", validation_curves, ["task", "initialization", "epoch", "val_n", "val_accuracy", "val_macro_f1", "fit_seconds"])
    six_errors = [row for row in predictions if row["task"] == "6class" and row["method"] == "official_fasttext" and row["initialization"] == "pretrained" and row["split"] == "test" and not row["correct"]]
    write_csv(RESULTS / "errors_6class_pretrained_fasttext.csv", six_errors, ["task", "method", "initialization", "split", "article_id", "true_label", "predicted_label", "correct", "confidence"])
    make_charts(metrics_rows, predictions, validation_curves)

    archive_stats = json.loads((DATA / "corpus_stats.json").read_text(encoding="utf-8"))
    run_meta = {
        "python": sys.version,
        "python_executable": sys.executable,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "classification_thread_environment": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "embedding_thread_environment": embedding_meta.get("thread_environment", "not_recorded_for_embedding_checkpoint"),
        "versions": {"gensim": gensim.__version__, "numpy": numpy.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__, "pyarrow": pyarrow.__version__, "jieba": jieba.__version__, "fasttext_wheel": importlib.metadata.version("fasttext-wheel")},
        "classification_corpus": archive_stats,
        "embedding_corpus": external_stats,
        "embedding_training": embedding_meta,
        "classification_rows": {task: {split: sum(row["label"] in TASKS[task] and row["split"] == split for row in rows) for split in ("train", "val", "test")} for task in TASKS},
        "metrics_rows": len(metrics_rows),
        "prediction_rows": len(predictions),
        "validation_curve_rows": len(validation_curves),
        "stage_timings_seconds": {"teacher_source_preprocessing": teacher_elapsed, "external_preprocessing": external_stats["preprocessing_seconds"], "embeddings_including_diagnostics": embedding_elapsed, "classification_candidates": classification_elapsed},
        "files": [str(path.relative_to(ROOT)) for path in sorted(RESULTS.iterdir()) if path.is_file()],
    }
    write_json(RESULTS / "run_metadata.json", run_meta)
    return run_meta


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    if not (ROOT / ARCHIVE_RELATIVE).is_file() or not (ROOT / RELATIVE_PARQUET).is_file():
        raise SystemExit("需要随项目提供 data/source 下固定教师语料归档及 data/external 下固定 revision 的 LR-Sum cmn_s train Parquet；见 README.md 与 data/SOURCE.md。")
    actual_external_hash = sha256_bytes((ROOT / RELATIVE_PARQUET).read_bytes())
    if actual_external_hash != EXPECTED_PARQUET_SHA256:
        raise SystemExit(f"LR-Sum Parquet SHA256 不匹配：{actual_external_hash}")
    diagnostic_path = Path(tempfile.gettempdir()) / f"pfr_fasttext_diagnostic_{os.getpid()}.log"
    diagnostic_handle = diagnostic_path.open("w", encoding="utf-8")
    faulthandler.enable(file=diagnostic_handle)
    faulthandler.dump_traceback_later(60, repeat=True, file=diagnostic_handle)
    completed = False
    try:
        print(f"[stage] diagnostics armed path={diagnostic_path}", flush=True)
        metadata = run_training()
        from audit import audit
        print("[stage] result-consistency audit start", flush=True)
        audit_started = time.perf_counter()
        audit_report = audit(ROOT)
        print(f"[stage] result-consistency audit done seconds={time.perf_counter() - audit_started:.2f} status={audit_report.get('status')}", flush=True)
        metadata["audit"] = audit_report
        metadata["files"] = sorted(
            str(path.relative_to(ROOT))
            for base in (DATA, MODELS, RESULTS)
            for path in base.rglob("*")
            if path.is_file()
        )
        write_json(RESULTS / "run_metadata.json", metadata)
        completed = True
    except ImportError:
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        diagnostic_handle.close()
        # fastText's Windows C++ layer cannot open the Chinese workspace path;
        # keep only its disposable UTF-8 train inputs in an ASCII temp folder.
        if SCRATCH.exists() and SCRATCH.resolve().parent == Path(tempfile.gettempdir()).resolve() and SCRATCH.name.startswith("pfr_fasttext_"):
            shutil.rmtree(SCRATCH)
        if completed and diagnostic_path.is_file():
            diagnostic_path.unlink()
    print(f"[done] metrics={RESULTS / 'metrics.csv'} predictions={RESULTS / 'predictions.csv'} audit={RESULTS / 'audit_report.json'} seed={SEED}; pythonhashseed={metadata['pythonhashseed']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

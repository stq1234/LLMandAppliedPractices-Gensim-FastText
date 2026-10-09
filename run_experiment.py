#!/usr/bin/env python3
"""Freeze a small People Daily sample and compare embedding and fastText classifiers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
import zipfile
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Iterable


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
SAMPLE_DATE = "20261009"
EXPECTED_ARCHIVE_SHA256 = "17474bbf2b360921f2eae0089fd83e22601e5a0649943c3f59403650997b4ed2"
EXPECTED_JAN_SHA256 = "41760f40739b0f301ab29e99ab14565e7cd4b06128eb3a6ac18bce277a6075a8"
JAN_MEMBER = "199801/199801.txt"
RECORD_RE = re.compile(r"^(\d{8})-(\d+)-(\d+)-(\d+)/([A-Za-z][A-Za-z0-9_-]*)$")
POS_RE = re.compile(r"^(.*)/([A-Za-z][A-Za-z0-9_-]*)$")
CHUNK_RE = re.compile(r"\]([A-Za-z][A-Za-z0-9_-]*)$")
TAG_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")

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
SELECTION_RULES = {
    "body_min_tokens": 80,
    "body_max_input_tokens": 500,
    "remove_pos": True,
    "remove_punctuation_pos": "w",
    "title_segment": "001",
    "keyword_matching": "exact token match after POS/chunk cleanup",
    "single_category_only": True,
    "same_category_multiple_keyword_tokens_allowed": True,
    "title_in_supervised_text": False,
    "per_category_target": 50,
    "fallback_per_category": 40,
    "selection_salt": f"{SAMPLE_DATE}:select:",
    "split_salt": f"{SAMPLE_DATE}:split:",
    "split_for_50": {"train": 30, "val": 10, "test": 10},
    "split_for_40": {"train": 24, "val": 8, "test": 8},
    "category_order": CATEGORY_ORDER,
    "categories": CATEGORY_KEYWORDS,
    "tasks": TASKS,
    "social_scope_note": "社会为广义民生/社会议题，和经济、政治、文化可能交叉；跨类别标题命中会被剔除，但弱标签仍可能有语义噪声。",
    "international_scope_note": "只用外国国名、联合国和外交相关词；不使用国际、亚洲、欧洲、中东等泛词。",
    "politics_scope_note": "使用核心政治机构/制度词；不使用国家、会议、领导、总统、中央、政策等泛词。",
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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def decode_pfr_token(piece: str) -> tuple[str, str]:
    chunk = CHUNK_RE.search(piece)
    if chunk:
        piece = piece[: chunk.start()]
    if piece.endswith("/%"):
        piece = piece[:-2]
    match = POS_RE.fullmatch(piece)
    if not match or not TAG_RE.fullmatch(match.group(2)):
        raise ValueError(f"无法解析 PFR 词性片段: {piece!r}")
    token, pos = match.groups()
    token = "".join(char for char in token if unicodedata.category(char) != "Cc")
    token = token.replace("[", "").replace("]", "")
    return token, pos


def extract_january(source: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = source.expanduser().resolve()
    if not source.is_file() or not zipfile.is_zipfile(source):
        raise ValueError(f"源文件不存在或不是 ZIP: {source}")
    archive_bytes = source.read_bytes()
    archive_hash = sha256_bytes(archive_bytes)
    if archive_hash != EXPECTED_ARCHIVE_SHA256:
        raise ValueError(f"ZIP 摘要不符合冻结源: expected={EXPECTED_ARCHIVE_SHA256}, actual={archive_hash}")

    groups: OrderedDict[str, dict[str, Any]] = OrderedDict()
    pos_counts: Counter[str] = Counter()
    segment_occurrences: dict[tuple[str, int], list[int]] = {}
    record_count = blank_count = physical_lines = punct_count = chunk_count = 0
    dates: set[str] = set()
    with zipfile.ZipFile(source) as archive:
        if JAN_MEMBER not in archive.namelist():
            raise ValueError(f"ZIP 中缺少一月成员 {JAN_MEMBER}")
        raw = archive.read(JAN_MEMBER)
    jan_hash = sha256_bytes(raw)
    if jan_hash != EXPECTED_JAN_SHA256:
        raise ValueError(f"一月成员摘要不符合冻结源: expected={EXPECTED_JAN_SHA256}, actual={jan_hash}")
    text = raw.decode("utf-8-sig", errors="strict")
    for line_no, line in enumerate(text.splitlines(), start=1):
        physical_lines += 1
        pieces = line.strip().split()
        if not pieces:
            blank_count += 1
            continue
        match = RECORD_RE.fullmatch(pieces[0])
        if not match:
            raise ValueError(f"{JAN_MEMBER}:{line_no} 记录编号格式错误: {pieces[0]!r}")
        date, edition, article, segment, _record_pos = match.groups()
        article_id = f"{date}-{edition}-{article}"
        dates.add(date)
        record_count += 1
        segments = groups.setdefault(article_id, {"segments": [], "date": date, "edition": edition, "article": article})["segments"]
        segment_id = int(segment)
        segment_occurrences.setdefault((article_id, segment_id), []).append(line_no)
        tokens: list[str] = []
        for piece in pieces[1:]:
            chunk_count += int(bool(CHUNK_RE.search(piece)))
            token, pos = decode_pfr_token(piece)
            pos_counts[pos] += 1
            if pos == "w":
                punct_count += 1
                continue
            if token:
                tokens.append(token)
        if not tokens:
            # Empty lines are handled before parsing; a punctuation-only record is not useful here.
            continue
        # Some source records reuse a segment number; retain every line in source order.
        segments.append({"segment_id": segment_id, "tokens": tokens})

    articles: list[dict[str, Any]] = []
    for article_id, raw_article in groups.items():
        segments: list[dict[str, Any]] = raw_article["segments"]
        title_record = next((row for row in segments if row["segment_id"] == 1), None)
        title_tokens = title_record["tokens"] if title_record else []
        body_tokens = [token for row in segments if row["segment_id"] != 1 for token in row["tokens"]]
        title = " ".join(title_tokens)
        body_text = " ".join(body_tokens)
        articles.append({
            "article_id": article_id,
            "date": raw_article["date"],
            "edition": raw_article["edition"],
            "article_number": raw_article["article"],
            "has_title_segment": bool(title_tokens),
            "segment_count": len(segments),
            "title_tokens": title_tokens,
            "title": title,
            "body_token_count": len(body_tokens),
            "body_tokens": body_tokens,
            "body_text": body_text,
        })

    duplicate_segments = [
        {"article_id": article_id, "segment_id": segment_id, "source_line_numbers": line_numbers}
        for (article_id, segment_id), line_numbers in segment_occurrences.items()
        if len(line_numbers) > 1
    ]
    stats = {
        "source_name": source.name,
        "archive_bytes": len(archive_bytes),
        "archive_sha256": archive_hash,
        "source_member": JAN_MEMBER,
        "source_member_bytes": len(raw),
        "source_member_sha256": jan_hash,
        "encoding": "utf-8-sig",
        "physical_lines": physical_lines,
        "blank_lines": blank_count,
        "records": record_count,
        "unique_dates": len(dates),
        "date_start": min(dates) if dates else "",
        "date_end": max(dates) if dates else "",
        "articles": len(articles),
        "articles_missing_title_segment": sum(not row["has_title_segment"] for row in articles),
        "articles_body_ge_80": sum(row["has_title_segment"] and row["body_token_count"] >= 80 for row in articles),
        "clean_nonpunct_tokens": sum(row["body_token_count"] + len(row["title_tokens"]) for row in articles),
        "pos_tags": dict(sorted(pos_counts.items())),
        "punctuation_tokens_dropped": punct_count,
        "chunk_markers_encountered": chunk_count,
        "duplicate_segment_keys": len(duplicate_segments),
        "duplicate_segment_records_extra": sum(len(row["source_line_numbers"]) - 1 for row in duplicate_segments),
        "duplicate_segment_examples": duplicate_segments,
    }
    return articles, stats


def classify_title(title_tokens: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    matched: dict[str, list[str]] = {}
    for category in CATEGORY_ORDER:
        found = sorted({token for token in title_tokens if token in CATEGORY_KEYWORDS[category]})
        if found:
            matched[category] = found
    return list(matched), matched


def freeze_dataset(source: Path) -> dict[str, Any]:
    articles, stats = extract_january(source)
    DATA.mkdir(parents=True, exist_ok=True)
    write_jsonl(DATA / "cleaned_january_articles.jsonl", articles)
    write_json(DATA / "selection_rules.json", SELECTION_RULES)

    candidate_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    for article in articles:
        if not article["has_title_segment"]:
            status = "missing_title"
            hits: list[str] = []
            matched: dict[str, list[str]] = {}
        else:
            hits, matched = classify_title(article["title_tokens"])
            if article["body_token_count"] < SELECTION_RULES["body_min_tokens"]:
                status = "short_body"
            elif not hits:
                status = "no_keyword_hit"
            elif len(hits) > 1:
                status = "cross_category_hit"
            else:
                status = "candidate"
                category = hits[0]
                candidate_rows.append({
                    "article_id": article["article_id"],
                    "label": category,
                    "title": article["title"],
                    "body_token_count": article["body_token_count"],
                    "matched_keywords": "|".join(matched[category]),
                    "selection_hash": salted_hash(SELECTION_RULES["selection_salt"], article["article_id"]),
                })
        status_counts[status] += 1
        audit_rows.append({
            "article_id": article["article_id"],
            "title": article["title"],
            "body_token_count": article["body_token_count"],
            "has_title_segment": article["has_title_segment"],
            "hit_categories": "|".join(hits),
            "matched_keywords_json": json.dumps(matched, ensure_ascii=False, sort_keys=True),
            "status": status,
        })

    # Keep a single six-class sample. The 2- and 4-class tasks are literal nested subsets.
    selected_rows: list[dict[str, Any]] = []
    target_per_class = 50 if all(sum(row["label"] == label for row in candidate_rows) >= 50 for label in CATEGORY_ORDER) else 40
    split_counts = SELECTION_RULES["split_for_50"] if target_per_class == 50 else SELECTION_RULES["split_for_40"]
    for label in CATEGORY_ORDER:
        pool = [row for row in candidate_rows if row["label"] == label]
        pool.sort(key=lambda row: (row["selection_hash"], row["article_id"]))
        minimum = SELECTION_RULES["fallback_per_category"]
        if len(pool) < minimum:
            raise ValueError(f"类别 {label} 只有 {len(pool)} 篇标题单命中且正文达标文章，低于最低 {minimum}；停止选样。")
        chosen = pool[:target_per_class]
        chosen.sort(key=lambda row: (salted_hash(SELECTION_RULES["split_salt"], row["article_id"]), row["article_id"]))
        for index, row in enumerate(chosen):
            if index < split_counts["train"]:
                split = "train"
            elif index < split_counts["train"] + split_counts["val"]:
                split = "val"
            else:
                split = "test"
            article = next(item for item in articles if item["article_id"] == row["article_id"])
            max_tokens = SELECTION_RULES["body_max_input_tokens"]
            body_tokens = article["body_tokens"][:max_tokens]
            selected_rows.append({
                "article_id": row["article_id"],
                "label": label,
                "split": split,
                "selection_hash": row["selection_hash"],
                "split_hash": salted_hash(SELECTION_RULES["split_salt"], row["article_id"]),
                "title": article["title"],
                "body_token_count": article["body_token_count"],
                "input_token_count": len(body_tokens),
                "body_text": " ".join(body_tokens),
                "matched_keywords": row["matched_keywords"],
            })

    selected_ids = {row["article_id"] for row in selected_rows}
    evaluation_rows = [row for row in selected_rows if row["split"] in {"val", "test"}]
    evaluation_ids = {row["article_id"] for row in evaluation_rows}
    article_by_id = {row["article_id"]: row for row in articles}
    eval_full_hashes = {sha256_text(article_by_id[row["article_id"]]["body_text"]) for row in evaluation_rows}
    eval_prefix_hashes = {sha256_text(article_by_id[row["article_id"]]["body_text"][:200]) for row in evaluation_rows}
    unsup_rows: list[dict[str, Any]] = []
    unsup_manifest: list[dict[str, Any]] = []
    for article in articles:
        body = article["body_text"]
        full_hash = sha256_text(body)
        prefix_hash = sha256_text(body[:200])
        if article["article_id"] in evaluation_ids:
            reason = "validation_or_test_id"
        elif full_hash in eval_full_hashes:
            reason = "duplicate_full_body_of_validation_or_test"
        elif prefix_hash in eval_prefix_hashes:
            reason = "duplicate_first_200_chars_of_validation_or_test"
        elif not article["body_tokens"]:
            reason = "empty_body_no_tokens"
        else:
            reason = "included"
            unsup_rows.append({"article_id": article["article_id"], "tokens": article["body_tokens"]})
        unsup_manifest.append({
            "article_id": article["article_id"],
            "body_token_count": article["body_token_count"],
            "full_body_sha256": full_hash,
            "first_200_chars_sha256": prefix_hash,
            "included": reason == "included",
            "reason": reason,
        })

    write_csv(DATA / "weak_label_candidates.csv", candidate_rows, ["article_id", "label", "title", "body_token_count", "matched_keywords", "selection_hash"])
    write_csv(DATA / "label_filter_audit.csv", audit_rows, ["article_id", "title", "body_token_count", "has_title_segment", "hit_categories", "matched_keywords_json", "status"])
    write_csv(DATA / "frozen_split.csv", selected_rows, ["article_id", "label", "split", "selection_hash", "split_hash", "title", "body_token_count", "input_token_count", "body_text", "matched_keywords"])
    write_csv(DATA / "unsupervised_corpus_manifest.csv", unsup_manifest, ["article_id", "body_token_count", "full_body_sha256", "first_200_chars_sha256", "included", "reason"])
    DATA.joinpath("unsupervised_corpus.txt").write_text("".join(" ".join(row["tokens"]) + "\n" for row in unsup_rows), encoding="utf-8", newline="\n")

    candidate_counts = {label: sum(row["label"] == label for row in candidate_rows) for label in CATEGORY_ORDER}
    selected_counts = {label: sum(row["label"] == label for row in selected_rows) for label in CATEGORY_ORDER}
    split_by_label = {label: {split: sum(row["label"] == label and row["split"] == split for row in selected_rows) for split in ("train", "val", "test")} for label in CATEGORY_ORDER}
    stats.update({
        "status_counts": dict(sorted(status_counts.items())),
        "single_category_candidates_under_six_class_rules": candidate_counts,
        "target_per_class": target_per_class,
        "selected_counts": selected_counts,
        "split_counts_by_label": split_by_label,
        "selected_articles": len(selected_rows),
        "unsupervised_articles_included": len(unsup_rows),
        "unsupervised_excluded_reason_counts": dict(Counter(row["reason"] for row in unsup_manifest if row["reason"] != "included")),
        "unsupervised_tokens": sum(len(row["tokens"]) for row in unsup_rows),
    })
    write_json(DATA / "corpus_stats.json", stats)
    return stats


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def maybe_save_gensim_model(model: Any, path: Path) -> dict[str, Any]:
    model.save(str(path))
    size = path.stat().st_size
    return {"path": str(path.relative_to(ROOT)), "bytes": size, "under_100_mib": size < 100 * 1024 * 1024}


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
    if saved_meta_path.is_file() and w2v_path.is_file() and ft_path.is_file():
        cached_meta = json.loads(saved_meta_path.read_text(encoding="utf-8"))
        if cached_meta.get("unsupervised_corpus_sha256") == corpus_hash:
            w2v = Word2Vec.load(str(w2v_path))
            ft = FastText.load(str(ft_path))
            return w2v, ft, cached_meta
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

    selected = read_csv(DATA / "frozen_split.csv")
    train_vocab = sorted({token for row in selected if row["split"] == "train" for token in row["body_text"].split()})
    pretrained_path = MODELS / "gensim_fasttext_train_vocab.vec"
    written = 0
    with pretrained_path.open("w", encoding="utf-8", newline="\n") as handle:
        known = [token for token in train_vocab if token in ft.wv.key_to_index]
        handle.write(f"{len(known)} {ft.vector_size}\n")
        for token in known:
            vector = ft.wv.get_vector(token)
            handle.write(token + " " + " ".join(f"{float(value):.8f}" for value in vector) + "\n")
            written += 1

    meta = {
        "w2v": {"seconds": w2v_seconds, "vocabulary_size": len(w2v.wv), "corpus_count": w2v.corpus_count, "artifact": w2v_meta, "params": {"sg": 1, "vector_size": 100, "window": 5, "min_count": 3, "epochs": 10, "workers": 1, "seed": SEED}},
        "gensim_fasttext": {"seconds": ft_seconds, "vocabulary_size": len(ft.wv), "corpus_count": ft.corpus_count, "artifact": ft_meta, "params": {"sg": 1, "vector_size": 100, "window": 5, "min_count": 3, "epochs": 10, "workers": 1, "seed": SEED, "min_n": 2, "max_n": 4, "bucket": 100000}},
        "pretrained_vectors": {"path": str(pretrained_path.relative_to(ROOT)), "bytes": pretrained_path.stat().st_size, "train_vocabulary_tokens": len(train_vocab), "exported_known_tokens": written, "oov_exported": 0},
        "unsupervised_corpus_sha256": sha256_bytes(corpus_path.read_bytes()),
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
    class_counts = [sum(row["label"] == label for row in split_rows) for label in CATEGORY_ORDER]
    split_counts = [sum(row["split"] == split for row in split_rows) for split in ("train", "val", "test")]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    axes[0].pie(class_counts, labels=[f"{label} ({count})" for label, count in zip(CATEGORY_ORDER, class_counts)], autopct="%1.1f%%", startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 1})
    axes[0].set_title("六类固定样本分布")
    axes[1].pie(split_counts, labels=[f"{name} ({count})" for name, count in zip(("训练", "验证", "测试"), split_counts)], autopct="%1.1f%%", startangle=90, colors=["#176B87", "#D97706", "#57A773"], wedgeprops={"edgecolor": "white", "linewidth": 1})
    axes[1].set_title("按类别哈希分层的固定拆分")
    figure.suptitle("实际筛选样本计数")
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

    curve_fig, curve_axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True, sharey=True)
    for axis, task in zip(curve_axes, tasks):
        for initialization in ("random", "pretrained"):
            subset = sorted((row for row in curves if row["task"] == task and row["initialization"] == initialization), key=lambda row: int(row["epoch"]))
            axis.plot([int(row["epoch"]) for row in subset], [float(row["val_macro_f1"]) for row in subset], marker="o", linewidth=2, label=initialization)
        axis.set_title(task)
        axis.set_xlabel("Epoch 候选值")
        axis.set_xticks([25, 100, 250])
        axis.grid(alpha=0.2)
    curve_axes[0].set_ylabel("验证集 Macro-F1")
    curve_axes[-1].legend(frameon=False)
    curve_fig.suptitle("官方 fastText 验证集选轮曲线")
    curve_fig.savefig(RESULTS / "fasttext_validation_curves.png", dpi=200, bbox_inches="tight")
    plt.close(curve_fig)


def run_training() -> dict[str, Any]:
    import fasttext
    import gensim
    import matplotlib
    import numpy
    import scipy
    import sklearn

    rows = load_samples()
    RESULTS.mkdir(parents=True, exist_ok=True)
    w2v, ft, embedding_meta = train_embeddings()
    embedding_diagnostics(w2v, ft)
    pretrained_path = MODELS / "gensim_fasttext_train_vocab.vec"
    metrics_rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    validation_curves: list[dict[str, Any]] = []
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
        "versions": {"gensim": gensim.__version__, "numpy": numpy.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__, "fasttext_wheel": "0.9.2"},
        "corpus": archive_stats,
        "embedding_training": embedding_meta,
        "classification_rows": {task: {split: sum(row["label"] in TASKS[task] and row["split"] == split for row in rows) for split in ("train", "val", "test")} for task in TASKS},
        "metrics_rows": len(metrics_rows),
        "prediction_rows": len(predictions),
        "validation_curve_rows": len(validation_curves),
        "files": [str(path.relative_to(ROOT)) for path in sorted(RESULTS.iterdir()) if path.is_file()],
    }
    write_json(RESULTS / "run_metadata.json", run_meta)
    return run_meta


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="本地 PeopleDaily1998 ZIP；指定时重建并冻结一月样本")
    args = parser.parse_args()
    if args.source:
        stats = freeze_dataset(args.source)
        print(f"[freeze] articles={stats['articles']} body_ge_80={stats['articles_body_ge_80']} candidates={stats['single_category_candidates_under_six_class_rules']} selected={stats['selected_counts']} splits={stats['split_counts_by_label']}", flush=True)
    elif not (DATA / "frozen_split.csv").is_file() or not (DATA / "unsupervised_corpus.txt").is_file():
        raise SystemExit("未找到冻结小样本。首次运行请提供 --source <人民日报1998_PFR.zip>。")
    try:
        metadata = run_training()
        from audit import audit
        audit_report = audit(ROOT)
        metadata["audit"] = audit_report
        metadata["files"] = sorted(
            str(path.relative_to(ROOT))
            for base in (DATA, MODELS, RESULTS)
            for path in base.rglob("*")
            if path.is_file()
        )
        write_json(RESULTS / "run_metadata.json", metadata)
    except ImportError:
        raise
    finally:
        # fastText's Windows C++ layer cannot open the Chinese workspace path;
        # keep only its disposable UTF-8 train inputs in an ASCII temp folder.
        if SCRATCH.exists() and SCRATCH.resolve().parent == Path(tempfile.gettempdir()).resolve() and SCRATCH.name.startswith("pfr_fasttext_"):
            import shutil
            shutil.rmtree(SCRATCH)
    print(f"[done] metrics={RESULTS / 'metrics.csv'} predictions={RESULTS / 'predictions.csv'} audit={RESULTS / 'audit_report.json'} seed={SEED}; pythonhashseed={metadata['pythonhashseed']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

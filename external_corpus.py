"""Build the independent LR-Sum Chinese training corpus used for embeddings."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


DATASET = "bltlab/lr-sum"
CONFIG = "cmn_s"
SPLIT = "train"
REVISION = "deb2e73ce53e42ceda31297a991afa138ee665a8"
RELATIVE_PARQUET = Path("data/external/lr_sum_cmn_s_train.parquet")
EXPECTED_PARQUET_SHA256 = "687db5968b23e492bfb904dc438e6ecff95f711ab93c8ef3d4d36e086d5e8630"
PARQUET_URL = (
    "https://huggingface.co/datasets/bltlab/lr-sum/resolve/"
    f"{REVISION}/cmn_s/train-00000-of-00001.parquet"
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize_text(value: str) -> str:
    """NFKC-normalize characters, discard controls, and collapse whitespace."""
    normalized = unicodedata.normalize("NFKC", value)
    kept = [char for char in normalized if unicodedata.category(char) not in {"Cc", "Cf", "Cs"}]
    return re.sub(r"\s+", " ", "".join(kept)).strip()


def canonical_body(value: str) -> str:
    """Make an article fingerprint insensitive to case, spacing, and punctuation."""
    normalized = normalize_text(value).casefold()
    return "".join(
        char for char in normalized
        if not char.isspace() and not unicodedata.category(char).startswith("P")
    )


def _digest_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _punctuation_only(token: str) -> bool:
    return bool(token) and all(unicodedata.category(char).startswith("P") for char in token)


def build_external_corpus(
    root: Path,
    source_path: Path | None = None,
    teacher_articles: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create the token corpus using only pinned LR-Sum text and check teacher overlap."""
    import jieba
    import pyarrow.parquet as parquet
    from teacher_corpus import build_teacher_corpus

    root = root.resolve()
    data_dir = root / "data"
    source = (source_path or (root / RELATIVE_PARQUET)).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"缺少冻结的 LR-Sum train Parquet：{source}")
    source_hash = sha256_bytes(source.read_bytes())
    if source_hash != EXPECTED_PARQUET_SHA256:
        raise ValueError(
            "LR-Sum 文件 SHA256 与固定 revision 不符："
            f"expected={EXPECTED_PARQUET_SHA256}, actual={source_hash}"
        )

    table = parquet.read_table(source)
    required_columns = {"id", "title", "summary", "text"}
    if not required_columns.issubset(table.column_names):
        raise ValueError(f"LR-Sum Parquet 缺少预期列：{table.column_names}")
    # This explicit projection is the information boundary: title, summary, and
    # any dataset labels are never accessed to construct embedding sentences.
    records = table.select(["id", "text"]).to_pylist()

    if teacher_articles is None:
        teacher_articles = build_teacher_corpus(root, write_outputs=False)["articles"]
    frozen_rows: list[dict[str, str]] = []
    with (data_dir / "frozen_split.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        frozen_rows = list(csv.DictReader(handle))
    eval_ids = {row["article_id"] for row in frozen_rows if row["split"] in {"val", "test"}}
    teacher_all_full: set[str] = set()
    teacher_all_prefix: set[str] = set()
    teacher_eval_full: set[str] = set()
    teacher_eval_prefix: set[str] = set()
    for article in teacher_articles:
        canonical = canonical_body(article["body_text"])
        if not canonical:
            continue
        full_hash = _digest_text(canonical)
        prefix_hash = _digest_text(canonical[:200])
        teacher_all_full.add(full_hash)
        teacher_all_prefix.add(prefix_hash)
        if article["article_id"] in eval_ids:
            teacher_eval_full.add(full_hash)
            teacher_eval_prefix.add(prefix_hash)

    jieba.setLogLevel(20)
    manifest: list[dict[str, Any]] = []
    cleaned: list[dict[str, Any]] = []
    corpus_lines: list[str] = []
    seen_external: set[str] = set()
    external_full_hashes: set[str] = set()
    external_prefix_hashes: set[str] = set()
    all_teacher_exact_matches = all_teacher_prefix_matches = 0
    eval_teacher_exact_matches = eval_teacher_prefix_matches = 0
    token_total = 0

    for row_index, record in enumerate(records):
        source_id = str(record.get("id") or "")
        normalized = normalize_text(record.get("text") or "")
        canonical = canonical_body(normalized)
        full_hash = _digest_text(canonical) if canonical else ""
        prefix_hash = _digest_text(canonical[:200]) if canonical else ""
        reason = "included"
        tokens: list[str] = []
        if not canonical:
            reason = "empty_body"
        elif full_hash in seen_external:
            reason = "duplicate_exact_external"
        else:
            seen_external.add(full_hash)
            external_full_hashes.add(full_hash)
            external_prefix_hashes.add(prefix_hash)
            if full_hash in teacher_all_full:
                all_teacher_exact_matches += 1
            elif prefix_hash in teacher_all_prefix:
                all_teacher_prefix_matches += 1
            if full_hash in teacher_eval_full:
                eval_teacher_exact_matches += 1
                reason = "duplicate_exact_teacher"
            elif prefix_hash in teacher_eval_prefix:
                eval_teacher_prefix_matches += 1
                reason = "duplicate_prefix_200_teacher"
            elif full_hash in teacher_all_full:
                reason = "duplicate_exact_teacher"
            elif prefix_hash in teacher_all_prefix:
                reason = "duplicate_prefix_200_teacher"
            else:
                tokens = [
                    token.strip()
                    for token in jieba.cut(normalized, cut_all=False)
                    if token.strip() and not _punctuation_only(token.strip())
                ]
                if not tokens:
                    reason = "empty_after_tokenization"

        included = reason == "included"
        manifest.append({
            "source_row": row_index,
            "source_id": source_id,
            "normalized_text_sha256": _digest_text(normalized) if normalized else "",
            "canonical_body_sha256": full_hash,
            "canonical_prefix_200_sha256": prefix_hash,
            "token_count": len(tokens),
            "included": included,
            "reason": reason,
            "text_only": True,
        })
        if included:
            token_total += len(tokens)
            line = " ".join(tokens)
            corpus_lines.append(line)
            cleaned.append({
                "source_row": row_index,
                "source_id": source_id,
                "normalized_text": normalized,
                "tokens": tokens,
            })

    data_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = data_dir / "unsupervised_corpus.txt"
    corpus_path.write_text("".join(line + "\n" for line in corpus_lines), encoding="utf-8", newline="\n")
    _write_csv(
        data_dir / "unsupervised_corpus_manifest.csv",
        manifest,
        ["source_row", "source_id", "normalized_text_sha256", "canonical_body_sha256", "canonical_prefix_200_sha256", "token_count", "included", "reason", "text_only"],
    )
    with (data_dir / "lr_sum_cmn_s_cleaned_train.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in cleaned:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    stats = {
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "revision": REVISION,
        "source_path": str(source.relative_to(root)) if source.is_relative_to(root) else source.name,
        "source_url": PARQUET_URL,
        "license": "CC BY 4.0",
        "source_sha256": source_hash,
        "parquet_bytes": source.stat().st_size,
        "source_rows": len(records),
        "included_articles": len(cleaned),
        "excluded_reasons": dict(sorted(Counter(row["reason"] for row in manifest if not row["included"]).items())),
        "included_tokens": token_total,
        "tokens_per_article_mean": token_total / len(cleaned) if cleaned else 0.0,
        "normalization": "Unicode NFKC, remove Cc/Cf/Cs controls, collapse whitespace, preserve punctuation for segmentation",
        "deduplication_fingerprint": "casefold + NFKC + remove whitespace and Unicode punctuation; exact text SHA-256",
        "cross_corpus_exact_matches_any_teacher": all_teacher_exact_matches,
        "cross_corpus_prefix_200_matches_any_teacher": all_teacher_prefix_matches,
        "cross_corpus_exact_matches_validation_test": eval_teacher_exact_matches,
        "cross_corpus_prefix_200_matches_validation_test": eval_teacher_prefix_matches,
        "tokenizer": "jieba",
        "jieba_version": jieba.__version__,
        "title_used": False,
        "summary_used": False,
        "source_labels_used": False,
        "embedding_corpus_sha256": sha256_bytes(corpus_path.read_bytes()),
        "embedding_manifest_sha256": sha256_bytes((data_dir / "unsupervised_corpus_manifest.csv").read_bytes()),
    }
    _write_json(data_dir / "external_corpus_stats.json", stats)
    return stats


def write_source_metadata(root: Path, teacher_source_lock: dict[str, Any]) -> dict[str, Any]:
    root = root.resolve()
    source = root / RELATIVE_PARQUET
    metadata = {
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "revision": REVISION,
        "source_url": PARQUET_URL,
        "source_sha256": sha256_bytes(source.read_bytes()),
        "source_bytes": source.stat().st_size,
        "license": "CC BY 4.0",
        "upstream_card": "https://huggingface.co/datasets/bltlab/lr-sum",
        "source_columns": ["id", "url", "title", "summary", "text"],
        "embedding_input_column": "text",
        "trace_column": "id",
        "excluded_columns_from_embedding_tokens": ["title", "summary", "url", "dataset labels"],
        "source_labels_used": False,
        "classification_source": teacher_source_lock,
    }
    _write_json(root / "data" / "external_source_lock.json", metadata)
    return metadata

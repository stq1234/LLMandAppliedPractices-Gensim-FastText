"""Prepare the fixed weak-label sample from the supplied 1998 H1 corpus."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import tarfile
import unicodedata
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any


ARCHIVE_RELATIVE = Path("data/source/teacher_1998_h1_gb18030.tar.xz")
ARCHIVE_SHA256 = "63ac4a05079b391662bde04240cf8995fa80075406ff10a748d7c8aeb146eb5c"
USER_RAR_NAME = "1998-上半年-带音(1).rar"
USER_RAR_BYTES = 12093296
USER_RAR_SHA256 = "30c4435bcce7d082c2175720a4549114f80656761df95d3c8b20f0ee48e5d7ab"

SOURCE_MEMBERS = {
    "1998-01-2003版-带音.txt": (5320005, "37b85555df2f72642ff9815ac01f51d4f50024d80ca0e2e8a6be6f93b1038639"),
    "1998-02-2003版-带音.txt": (5441098, "a050351664f459aed2247bc45cd46c25da2ce893e11c44a9dea06a54529f19d8"),
    "1998-03-2003版-带音.txt": (5951910, "2f9d8ecfd92eec1b8a7009bd146da5faad218a34ae17fe460944bd772b2e97db"),
    "1998-04-2003版-带音.txt": (6107764, "43462d53f2f7603168745af3200a9e6fe883ddcdf18618280358499e0ffc3fca"),
    "1998-05-2003版-带音.txt": (5920187, "04ae37614a55c2418bcb5c7697c4980afa75a74734e930f06bee6e83022d0aea"),
    "1998-06-2003版-带音.txt": (5946378, "f18f8f3d6e886644baf8306e1f08a31e3a8d390709782f913ae1ea12f3e97c5f"),
}

SELECTION_SALT = "20261015:teacher-h1:select:"
SPLIT_SALT = "20261015:teacher-h1:split:"
TARGET_PER_CLASS = 200
MIN_BODY_TOKENS = 80
MAX_INPUT_TOKENS = 500
SPLIT_NAMES = ("train", "val", "test")
SPLIT_PROPORTIONS_200 = {"train": 140, "val": 30, "test": 30}

RECORD_PREFIX_BYTES = re.compile(rb"^(\d{8}-\d{2}-\d{3})-(\d{3})(?:\s+|$)")
RECORD_LINE = re.compile(r"^(\d{8}-\d{2}-\d{3})-(\d{3})\s+(.*)$")
PINYIN_ANNOTATION = re.compile(r"\{[^{}]*\}")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256(value.encode("utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _clean_tokens(text: str) -> list[str]:
    """Remove inline pinyin and punctuation-only tokens without retokenizing."""
    cleaned = PINYIN_ANNOTATION.sub("", text)
    output: list[str] = []
    for token in cleaned.split():
        if not token:
            continue
        if all(unicodedata.category(char)[0] in {"P", "S", "Z", "C"} for char in token):
            continue
        output.append(token)
    return output


def _weak_label_cue_overlap(selected_rows: list[dict[str, Any]], categories: list[str]) -> dict[str, Any]:
    """Summarize exact title-label cue tokens present in the classifier body input."""
    by_split: dict[str, dict[str, Any]] = {}
    by_split_and_label: dict[str, dict[str, Any]] = {}
    for split in SPLIT_NAMES:
        split_rows = [row for row in selected_rows if row["split"] == split]
        overlaps: list[tuple[dict[str, Any], list[str]]] = []
        for row in split_rows:
            title_cues: dict[str, set[str]] = defaultdict(set)
            for entry in row["matched_keywords"].split(";"):
                if not entry:
                    continue
                category, separator, cue_text = entry.partition(":")
                if not separator or category not in categories:
                    raise ValueError(f"弱标签关键词格式错误：{row['article_id']}")
                title_cues[category].update(cue for cue in cue_text.split("|") if cue)
            if set(title_cues) != {row["label"]}:
                raise ValueError(f"弱标签候选类别不唯一：{row['article_id']}")
            body_tokens = set(row["body_text"].split())
            overlaps.append((row, sorted(title_cues[row["label"]] & body_tokens)))

        hit_count = sum(bool(found) for _, found in overlaps)
        by_split[split] = {
            "articles": len(split_rows),
            "articles_with_any_title_label_keyword_in_classifier_body": hit_count,
            "fraction": hit_count / len(split_rows) if split_rows else None,
            "percent": 100 * hit_count / len(split_rows) if split_rows else None,
        }
        for label in sorted({row["label"] for row in split_rows}):
            label_rows = [(row, found) for row, found in overlaps if row["label"] == label]
            label_hits = sum(bool(found) for _, found in label_rows)
            by_split_and_label[f"{split}:{label}"] = {
                "articles": len(label_rows),
                "articles_with_any_title_label_keyword_in_classifier_body": label_hits,
                "fraction": label_hits / len(label_rows) if label_rows else None,
                "article_ids_with_overlap": [row["article_id"] for row, found in label_rows if found],
                "matched_keywords_found_in_body": {row["article_id"]: found for row, found in label_rows if found},
            }

    return {
        "purpose": "Audit whether at least one exact weak-label keyword selected from an article title is present as an exact source token in that article’s actual classifier body input.",
        "label_source": "Exact token match against the title keyword list in data/selection_rules.json; these are weak labels, not human gold labels.",
        "feature_source": "body_text in data/frozen_split.csv, already truncated to the first at most 500 source tokens; title is not part of the classifier input.",
        "match_definition": "For each article, intersect the deduplicated title-hit keyword tokens listed for its weak-label category in matched_keywords with the token set of body_text. Count an article once if the intersection is nonempty.",
        "this_is_split_leakage": False,
        "interpretation": "High overlap means model scores may rely on direct lexical cues correlated with the title-derived weak labels; it does not establish topic understanding when those cues are absent.",
        "source_file": "data/frozen_split.csv",
        "source_rows": len(selected_rows),
        "by_split": by_split,
        "by_split_and_label": by_split_and_label,
    }


def _load_source_members(archive_path: Path) -> tuple[dict[str, bytes], dict[str, dict[str, Any]]]:
    archive_bytes = archive_path.read_bytes()
    archive_hash = _sha256(archive_bytes)
    if archive_hash != ARCHIVE_SHA256:
        raise ValueError(f"教师语料压缩包摘要不匹配：{archive_hash} != {ARCHIVE_SHA256}")

    member_bytes: dict[str, bytes] = {}
    member_stats: dict[str, dict[str, Any]] = {}
    with tarfile.open(archive_path, mode="r:xz") as archive:
        regular_files = {member.name: member for member in archive.getmembers() if member.isfile()}
        if set(regular_files) != set(SOURCE_MEMBERS):
            raise ValueError(f"教师语料压缩包成员不符：{sorted(regular_files)}")
        for name, (expected_bytes, expected_hash) in SOURCE_MEMBERS.items():
            member = regular_files[name]
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"无法读取教师语料成员：{name}")
            content = stream.read()
            actual_hash = _sha256(content)
            if len(content) != expected_bytes or actual_hash != expected_hash:
                raise ValueError(f"教师语料成员摘要不匹配：{name} ({len(content)}, {actual_hash})")
            member_bytes[name] = content
            member_stats[name] = {"bytes": len(content), "sha256": actual_hash}
    return member_bytes, {
        "archive_bytes": len(archive_bytes),
        "archive_sha256": archive_hash,
        "members": member_stats,
    }


def build_teacher_corpus(root: Path, write_outputs: bool = True) -> dict[str, Any]:
    """Parse the six source files, freeze a balanced sample, and return full bodies."""
    root = root.resolve()
    data_dir = root / "data"
    archive_path = root / ARCHIVE_RELATIVE
    member_bytes, archive_stats = _load_source_members(archive_path)
    rules_path = data_dir / "selection_rules.json"
    rules = json.loads(rules_path.read_text(encoding="utf-8"))
    categories: list[str] = rules["category_order"]
    keywords: dict[str, list[str]] = rules["categories"]
    tasks: dict[str, list[str]] = rules["tasks"]

    records: OrderedDict[str, dict[str, Any]] = OrderedDict()
    decode_errors: list[dict[str, Any]] = []
    invalid_articles: set[str] = set()
    same_id_duplicate_rows = 0
    conflicting_record_ids: set[str] = set()
    malformed_nonempty_rows: list[dict[str, Any]] = []
    blank_lines = 0
    physical_lines = 0
    source_member_line_counts: dict[str, int] = {}

    for member_name, content in member_bytes.items():
        offset = 0
        line_count = 0
        for line_number, raw_line in enumerate(content.splitlines(keepends=True), start=1):
            line_count += 1
            physical_lines += 1
            line_bytes = raw_line.rstrip(b"\r\n")
            try:
                line = line_bytes.decode("gb18030", errors="strict")
            except UnicodeDecodeError as error:
                header = RECORD_PREFIX_BYTES.match(line_bytes)
                article_id = header.group(1).decode("ascii") if header else ""
                segment_id = header.group(2).decode("ascii") if header else ""
                if not article_id:
                    raise ValueError(f"GB18030 解码错误且无法识别文章 ID：{member_name}:{line_number}") from error
                invalid_articles.add(article_id)
                decode_errors.append({
                    "source_member": member_name,
                    "line_number": line_number,
                    "absolute_byte_offset": offset + error.start,
                    "article_id": article_id,
                    "segment_id": segment_id,
                    "record_id": f"{article_id}-{segment_id}",
                    "invalid_byte_hex": line_bytes[error.start:error.end].hex(),
                })
                offset += len(raw_line)
                continue

            if not line.strip():
                blank_lines += 1
                offset += len(raw_line)
                continue
            match = RECORD_LINE.match(line.strip())
            if not match:
                malformed_nonempty_rows.append({"source_member": member_name, "line_number": line_number, "prefix": line[:120]})
                offset += len(raw_line)
                continue
            article_id, segment_id, body = match.groups()
            record_id = f"{article_id}-{segment_id}"
            body = body.strip()
            location = {"source_member": member_name, "line_number": line_number}
            if record_id in records:
                prior = records[record_id]
                prior["source_locations"].append(location)
                if prior["body"] == body:
                    same_id_duplicate_rows += 1
                else:
                    prior["conflict"] = True
                    prior["alternates"].append({**location, "body_sha256": _sha256_text(body)})
                    conflicting_record_ids.add(record_id)
            else:
                records[record_id] = {
                    "article_id": article_id,
                    "segment_id": segment_id,
                    "body": body,
                    "source_locations": [location],
                    "conflict": False,
                    "alternates": [],
                }
            offset += len(raw_line)
        source_member_line_counts[member_name] = line_count

    if malformed_nonempty_rows:
        raise ValueError(f"教师语料中存在无法解析的非空记录：{malformed_nonempty_rows[:5]}")

    records_by_article: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for record in records.values():
        records_by_article.setdefault(record["article_id"], []).append(record)

    conflicting_articles = {
        record_id.rsplit("-", 1)[0] for record_id in conflicting_record_ids
    }
    decode_error_articles = set(invalid_articles)
    seen_body_hashes: dict[str, str] = {}
    articles: list[dict[str, Any]] = []
    article_audit: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    body_duplicate_count = 0
    missing_title_count = 0
    empty_body_count = 0
    pinyin_annotations = 0

    for article_id, article_records in records_by_article.items():
        source_members = sorted({location["source_member"] for record in article_records for location in record["source_locations"]})
        first_member = source_members[0] if source_members else ""
        title_records = [record for record in article_records if record["segment_id"] == "001"]
        if article_id in decode_error_articles:
            status = "strict_gb18030_decode_error"
            title_tokens: list[str] = []
            body_tokens: list[str] = []
            canonical = ""
            duplicate_of = ""
        elif article_id in conflicting_articles or any(record["conflict"] for record in article_records):
            status = "conflicting_segment_content"
            title_tokens = []
            body_tokens = []
            canonical = ""
            duplicate_of = ""
            for record in article_records:
                if record["conflict"]:
                    first = record["source_locations"][0]
                    for alternate in record["alternates"]:
                        conflict_rows.append({
                            "article_id": article_id,
                            "record_id": f"{article_id}-{record['segment_id']}",
                            "first_member": first["source_member"],
                            "first_line_number": first["line_number"],
                            "first_body_sha256": _sha256_text(record["body"]),
                            "alternate_member": alternate["source_member"],
                            "alternate_line_number": alternate["line_number"],
                            "alternate_body_sha256": alternate["body_sha256"],
                        })
        elif not title_records:
            status = "missing_title_segment"
            title_tokens = []
            body_tokens = []
            canonical = ""
            duplicate_of = ""
            missing_title_count += 1
        else:
            title_body = title_records[0]["body"]
            pinyin_annotations += len(PINYIN_ANNOTATION.findall(title_body))
            title_tokens = _clean_tokens(title_body)
            body_records = sorted((record for record in article_records if record["segment_id"] != "001"), key=lambda record: record["segment_id"])
            body_tokens = []
            for record in body_records:
                pinyin_annotations += len(PINYIN_ANNOTATION.findall(record["body"]))
                body_tokens.extend(_clean_tokens(record["body"]))
            canonical = "".join(body_tokens).casefold()
            if not body_tokens:
                status = "empty_body"
                empty_body_count += 1
                duplicate_of = ""
            else:
                body_hash = _sha256_text(canonical)
                duplicate_of = seen_body_hashes.get(body_hash, "")
                if duplicate_of:
                    status = "duplicate_canonical_body"
                    body_duplicate_count += 1
                else:
                    seen_body_hashes[body_hash] = article_id
                    status = "pending_label_filter"

        canonical_sha256 = _sha256_text(canonical) if canonical else ""
        title_hits = {
            category: sorted(set(token for token in title_tokens if token in keywords[category]))
            for category in categories
        }
        matched_categories = [category for category in categories if title_hits[category]]
        matched_keywords = ";".join(
            f"{category}:{'|'.join(title_hits[category])}" for category in matched_categories
        )
        body_token_count = len(body_tokens)
        if status == "pending_label_filter":
            if body_token_count < MIN_BODY_TOKENS:
                status = "short_body"
            elif not matched_categories:
                status = "no_keyword_hit"
            elif len(matched_categories) > 1:
                status = "cross_category_hit"
            else:
                status = "candidate"

        article_audit.append({
            "article_id": article_id,
            "source_members": ";".join(source_members),
            "segment_count": len(article_records),
            "body_token_count": body_token_count,
            "body_sha256": canonical_sha256,
            "title": " ".join(title_tokens),
            "matched_categories": ";".join(matched_categories),
            "matched_keywords": matched_keywords,
            "status": status,
            "duplicate_of": duplicate_of,
        })
        if status in {"pending_label_filter", "candidate", "short_body", "no_keyword_hit", "cross_category_hit"}:
            articles.append({
                "article_id": article_id,
                "source_members": source_members,
                "title_tokens": title_tokens,
                "title": " ".join(title_tokens),
                "body_tokens": body_tokens,
                "body_text": " ".join(body_tokens),
                "body_token_count": body_token_count,
                "canonical_body": canonical,
                "body_sha256": canonical_sha256,
                "matched_categories": matched_categories,
                "title_hits": title_hits,
                "matched_keywords": matched_keywords,
                "status": status,
            })

    candidate_articles = [article for article in articles if article["status"] == "candidate"]
    candidate_counts = Counter(article["matched_categories"][0] for article in candidate_articles)
    if set(candidate_counts) != set(categories):
        raise ValueError(f"弱标签候选类别缺失：{dict(candidate_counts)}")
    minimum_candidate_count = min(candidate_counts.values())
    target = min(TARGET_PER_CLASS, minimum_candidate_count)
    if target < 50:
        raise ValueError(f"最小类别候选数 {minimum_candidate_count} 少于 50，无法沿用六类任务。")
    split_counts = (
        dict(SPLIT_PROPORTIONS_200)
        if target == TARGET_PER_CLASS
        else {
            "train": int(target * 0.70),
            "val": int(target * 0.15),
            "test": target - int(target * 0.70) - int(target * 0.15),
        }
    )

    candidate_rows: list[dict[str, Any]] = []
    frozen_rows: list[dict[str, Any]] = []
    candidate_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for article in candidate_articles:
        label = article["matched_categories"][0]
        candidate = {
            **article,
            "label": label,
            "selection_hash": _sha256_text(SELECTION_SALT + article["article_id"]),
            "split_hash": _sha256_text(SPLIT_SALT + article["article_id"]),
        }
        candidate_by_label[label].append(candidate)
        candidate_rows.append({
            "article_id": article["article_id"],
            "label": label,
            "source_members": ";".join(article["source_members"]),
            "title": article["title"],
            "body_token_count": article["body_token_count"],
            "body_sha256": article["body_sha256"],
            "selection_hash": candidate["selection_hash"],
            "matched_keywords": article["matched_keywords"],
        })

    for label in categories:
        ordered = sorted(candidate_by_label[label], key=lambda row: (row["selection_hash"], row["article_id"]))
        chosen = ordered[:target]
        split_order = sorted(chosen, key=lambda row: (row["split_hash"], row["article_id"]))
        boundaries = {
            "train": split_counts["train"],
            "val": split_counts["train"] + split_counts["val"],
        }
        for index, row in enumerate(split_order):
            split = "train" if index < boundaries["train"] else "val" if index < boundaries["val"] else "test"
            tokens = row["body_tokens"][:MAX_INPUT_TOKENS]
            frozen_rows.append({
                "article_id": row["article_id"],
                "label": label,
                "split": split,
                "selection_hash": row["selection_hash"],
                "split_hash": row["split_hash"],
                "title": row["title"],
                "body_token_count": row["body_token_count"],
                "input_token_count": len(tokens),
                "body_text": " ".join(tokens),
                "matched_keywords": row["matched_keywords"],
                "canonical_body_sha256": row["body_sha256"],
                "source_members": ";".join(row["source_members"]),
            })

    frozen_rows.sort(key=lambda row: (categories.index(row["label"]), SPLIT_NAMES.index(row["split"]), row["selection_hash"]))

    selection_note = (
        "Each class has at least 200 eligible articles; choose 200 per class by salted SHA-256 order."
        if target == TARGET_PER_CLASS
        else f"Minimum eligible class count is {minimum_candidate_count} (<200); use a common target of {target} per class."
    )
    rules.update({
        "per_category_target": target,
        "minimum_candidate_count": minimum_candidate_count,
        "split_per_class": split_counts,
        "selection_salt": SELECTION_SALT,
        "split_salt": SPLIT_SALT,
        "body_min_tokens": MIN_BODY_TOKENS,
        "body_max_input_tokens": MAX_INPUT_TOKENS,
        "title_in_supervised_text": False,
        "remove_pos": False,
        "remove_pinyin_annotations": True,
        "remove_punctuation_tokens": True,
        "source_encoding": "GB18030 strict; exclude an entire article if any record cannot decode",
        "source_deduplication": "collapse exact same-ID/same-body records; exclude conflicting same-ID article; deduplicate canonical bodies before label filtering and splitting",
        "sample_note": selection_note,
    })

    stats = {
        "source_dataset": "User-provided 1998 first-half teacher corpus with pronunciation annotations",
        "source_encoding": "GB18030 strict",
        "source_rar_name": USER_RAR_NAME,
        "source_rar_bytes": USER_RAR_BYTES,
        "source_rar_sha256": USER_RAR_SHA256,
        "source_archive": str(ARCHIVE_RELATIVE).replace("\\", "/"),
        **archive_stats,
        "source_member_line_counts": source_member_line_counts,
        "physical_lines": physical_lines,
        "blank_lines": blank_lines,
        "malformed_nonempty_lines": len(malformed_nonempty_rows),
        "unique_segment_ids": len(records),
        "same_id_same_body_duplicate_rows_dropped": same_id_duplicate_rows,
        "same_id_conflicting_segment_ids": sorted(conflicting_record_ids),
        "conflicting_articles_dropped": len(conflicting_articles),
        "strict_gb18030_error_rows": len(decode_errors),
        "strict_gb18030_error_articles_dropped": len(decode_error_articles),
        "strict_gb18030_error_article_ids": sorted(decode_error_articles),
        "article_id_groups": len(records_by_article),
        "missing_title_articles_dropped": missing_title_count,
        "canonical_duplicate_body_articles_dropped": body_duplicate_count,
        "empty_body_articles_dropped": empty_body_count,
        "canonical_articles_after_integrity_and_dedup": len(articles),
        "pinyin_annotations_removed": pinyin_annotations,
        "body_min_tokens": MIN_BODY_TOKENS,
        "max_classifier_input_tokens": MAX_INPUT_TOKENS,
        "candidate_counts_by_label": {label: candidate_counts[label] for label in categories},
        "cross_category_title_hits": sum(row["status"] == "cross_category_hit" for row in article_audit),
        "no_keyword_title_hits": sum(row["status"] == "no_keyword_hit" for row in article_audit),
        "short_body_articles": sum(row["status"] == "short_body" for row in article_audit),
        "minimum_candidate_count": minimum_candidate_count,
        "selected_per_class": target,
        "selected_articles": len(frozen_rows),
        "split_per_class": split_counts,
        "selection_note": selection_note,
        "title_used_for_label_only": True,
        "title_in_classifier_input": False,
        "source_id_fields": "YYYYMMDD-page-article-segment",
    }
    member_stat_lookup = archive_stats["members"]
    source_lock = {
        "user_supplied_archive": USER_RAR_NAME,
        "user_supplied_archive_sha256": USER_RAR_SHA256,
        "user_supplied_archive_bytes": USER_RAR_BYTES,
        "derived_source_archive": str(ARCHIVE_RELATIVE).replace("\\", "/"),
        "derived_source_archive_sha256": archive_stats["archive_sha256"],
        "derived_source_archive_bytes": archive_stats["archive_bytes"],
        "encoding": "GB18030 strict per record",
        "members": member_stat_lookup,
        "selected_source_window": "1998-01 through 1998-06",
    }

    result = {
        "articles": articles,
        "candidate_articles": candidate_articles,
        "selected_rows": frozen_rows,
        "candidate_rows": candidate_rows,
        "article_audit_rows": article_audit,
        "decode_errors": decode_errors,
        "conflict_rows": conflict_rows,
        "stats": stats,
        "source_lock": source_lock,
        "rules": rules,
    }

    if write_outputs:
        data_dir.mkdir(parents=True, exist_ok=True)
        _write_json(rules_path, rules)
        _write_json(data_dir / "corpus_stats.json", stats)
        _write_json(data_dir / "teacher_source_lock.json", source_lock)
        _write_csv(
            data_dir / "frozen_split.csv",
            frozen_rows,
            ["article_id", "label", "split", "selection_hash", "split_hash", "title", "body_token_count", "input_token_count", "body_text", "matched_keywords", "canonical_body_sha256", "source_members"],
        )
        _write_json(root / "results" / "weak_label_cue_overlap.json", _weak_label_cue_overlap(frozen_rows, categories))
        _write_csv(
            data_dir / "weak_label_candidates.csv",
            candidate_rows,
            ["article_id", "label", "source_members", "title", "body_token_count", "body_sha256", "selection_hash", "matched_keywords"],
        )
        _write_csv(
            data_dir / "label_filter_audit.csv",
            article_audit,
            ["article_id", "source_members", "segment_count", "body_token_count", "body_sha256", "title", "matched_categories", "matched_keywords", "status", "duplicate_of"],
        )
        _write_csv(
            data_dir / "teacher_decode_errors.csv",
            decode_errors,
            ["source_member", "line_number", "absolute_byte_offset", "article_id", "segment_id", "record_id", "invalid_byte_hex"],
        )
        _write_csv(
            data_dir / "teacher_conflicting_records.csv",
            conflict_rows,
            ["article_id", "record_id", "first_member", "first_line_number", "first_body_sha256", "alternate_member", "alternate_line_number", "alternate_body_sha256"],
        )
    return result

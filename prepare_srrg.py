#!/usr/bin/env python
"""Build local FinAl JSONL manifests from gated SRRG and MIMIC-CXR-JPG."""

import argparse
import json
import re
from pathlib import Path

import pandas as pd


VALID_VIEWS = {"PA", "AP"}
SPLIT_MAP = {"train": "train", "validate": "valid", "test": "valid", "valid": "valid"}
VALID_FINDINGS_HEADERS = {
    "lungs and airways",
    "pleura",
    "cardiovascular",
    "hila and mediastinum",
    "tubes, catheters, and support devices",
    "musculoskeletal and chest wall",
    "abdominal",
    "other",
}
WORD_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def _clean_id(value):
    value = str(value).strip()
    return value[:-2] if value.endswith(".0") else value


def normalize_srrg_path(value):
    value = str(value).replace("\\", "/").strip()
    marker = "files/"
    position = value.find(marker)
    if position < 0:
        return value.lstrip("/")
    return value[position:]


def build_metadata_index(metadata_df, split_df):
    required_metadata = {"dicom_id", "subject_id", "study_id", "ViewPosition"}
    required_split = {"dicom_id", "subject_id", "study_id", "split"}
    missing_metadata = required_metadata - set(metadata_df.columns)
    missing_split = required_split - set(split_df.columns)
    if missing_metadata:
        raise ValueError("metadata CSV is missing columns: {}".format(sorted(missing_metadata)))
    if missing_split:
        raise ValueError("split CSV is missing columns: {}".format(sorted(missing_split)))

    metadata = metadata_df[list(required_metadata)].copy()
    split = split_df[list(required_split)].copy()
    for frame in (metadata, split):
        for column in ("dicom_id", "subject_id", "study_id"):
            frame[column] = frame[column].map(_clean_id)
    metadata = metadata[metadata["ViewPosition"].astype(str).str.upper().isin(VALID_VIEWS)]
    merged = metadata.merge(split, on=["dicom_id", "subject_id", "study_id"], how="inner")

    index = {}
    for row in merged.itertuples(index=False):
        split_name = SPLIT_MAP.get(str(row.split).strip().lower())
        if split_name is None:
            continue
        view = str(row.ViewPosition).upper()
        subject = _clean_id(row.subject_id)
        study = _clean_id(row.study_id)
        dicom = _clean_id(row.dicom_id)
        if study.startswith("s"):
            study = study[1:]
        relative = "files/p{}/p{}/s{}/{}.jpg".format(subject[:2], subject, study, dicom)
        item = {
            "image_path": "2.1.0/{}".format(relative),
            "split": split_name,
            "view": view,
        }
        index[relative] = item
        index[dicom] = item
    return index


def _report_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, dict):
        parts = [_report_text(part) for part in value.values()]
        return " ".join(part for part in parts if part)
    if isinstance(value, (list, tuple)):
        parts = [_report_text(part) for part in value]
        return " ".join(part for part in parts if part)
    return " ".join(str(value).split())


def _normalize_utterance(value):
    """Match the punctuation-free, case-preserving SRRG leaf representation."""
    return " ".join(WORD_PATTERN.findall(str(value)))


def _section_index(lines, name):
    expected = name.casefold().rstrip(":")
    for index, line in enumerate(lines):
        if line.casefold().rstrip(":").strip() == expected:
            return index
    return None


def extract_srrg_leaves(structured_report):
    """Extract findings utterances followed by impression utterances."""
    if not isinstance(structured_report, str):
        return []
    lines = [line.strip() for line in structured_report.splitlines() if line.strip()]
    findings_index = _section_index(lines, "findings")
    impression_index = _section_index(lines, "impression")
    if findings_index is None:
        return []

    findings_end = impression_index if impression_index is not None else len(lines)
    findings = []
    current_organ = None
    for line in lines[findings_index + 1 : findings_end]:
        potential_header = re.sub(r"^[^\w]*", "", line).rstrip(":").strip()
        if potential_header.casefold() in VALID_FINDINGS_HEADERS:
            current_organ = potential_header.casefold()
        elif line.startswith("-") and current_organ is not None:
            utterance = _normalize_utterance(line[1:].strip())
            if utterance:
                findings.append(utterance)
        else:
            current_organ = None

    impressions = []
    if impression_index is not None:
        for line in lines[impression_index + 1 :]:
            line = re.sub(r"^\d+\.\s*", "", line)
            utterance = _normalize_utterance(line)
            if utterance:
                impressions.append(utterance)
    return findings + impressions


def extract_report(record, report_field="structured_report", report_mode="final-leaves"):
    value = record.get(report_field)
    if report_mode == "final-leaves":
        if report_field != "structured_report":
            raise ValueError("final-leaves mode requires --report-field structured_report")
        return " ".join(extract_srrg_leaves(value))
    if report_mode == "hf-native":
        return _report_text(value)
    raise ValueError("unknown report mode: {}".format(report_mode))


def build_manifest_records(
    srrg_records,
    metadata_index,
    report_field="structured_report",
    report_mode="final-leaves",
):
    output = {}
    stats = {
        "srrg_rows": 0,
        "image_paths": 0,
        "unmatched": 0,
        "empty_report": 0,
        "duplicates": 0,
        "conflicts": 0,
    }
    for record in srrg_records:
        stats["srrg_rows"] += 1
        report = extract_report(
            record,
            report_field=report_field,
            report_mode=report_mode,
        )
        image_paths = record.get("image_paths") or record.get("image_path") or []
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        if not report:
            stats["empty_report"] += len(image_paths)
            continue
        for raw_path in image_paths:
            stats["image_paths"] += 1
            relative = normalize_srrg_path(raw_path)
            dicom = Path(relative).stem
            metadata = metadata_index.get(relative) or metadata_index.get(dicom)
            if metadata is None:
                stats["unmatched"] += 1
                continue
            item = {
                "image_path": metadata["image_path"],
                "report": report,
                "split": metadata["split"],
            }
            previous = output.get(metadata["image_path"])
            if previous is not None:
                stats["duplicates"] += 1
                stats["conflicts"] += previous["report"] != report
            output[metadata["image_path"]] = item
    records = sorted(output.values(), key=lambda item: (item["split"], item["image_path"]))
    stats["kept"] = len(records)
    return records, stats


def _find_mimic_csv(mimic_root, filename):
    root = Path(mimic_root)
    candidates = [root / filename, root / "2.1.0" / filename]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("could not find {} below {}".format(filename, root))


def write_manifests(records, manifest_dir, overwrite=False):
    manifest_dir = Path(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    by_split = {"train": [], "valid": []}
    for record in records:
        by_split.setdefault(record["split"], []).append(record)
    for split, split_records in by_split.items():
        path = manifest_dir / "{}.jsonl".format(split)
        if path.exists() and not overwrite:
            raise FileExistsError("{} exists; pass --overwrite to replace it".format(path))
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for record in split_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic-root", required=True, help="MIMIC-CXR-JPG root (or its 2.1.0 directory)")
    parser.add_argument("--manifest-dir", required=True, help="Private output directory for JSONL manifests")
    parser.add_argument("--metadata-csv", help="Override mimic-cxr-2.0.0-metadata.csv.gz")
    parser.add_argument("--split-csv", help="Override mimic-cxr-2.0.0-split.csv.gz")
    parser.add_argument("--dataset-id", default="StanfordAIMI/srrg_findings")
    parser.add_argument(
        "--dataset-revision",
        help="Optional Hugging Face dataset revision (commit, tag, or branch)",
    )
    parser.add_argument(
        "--report-mode",
        choices=("final-leaves", "hf-native"),
        default="final-leaves",
        help=(
            "final-leaves reproduces the FinAl caption assembly; hf-native uses "
            "the selected Hugging Face report field without leaf extraction"
        ),
    )
    parser.add_argument(
        "--report-field",
        default="structured_report",
        help=(
            "SRRG field to use. final-leaves requires structured_report; hf-native "
            "whitespace-normalizes the selected field directly."
        ),
    )
    parser.add_argument("--cache-dir", help="Private Hugging Face cache directory")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    metadata_path = Path(args.metadata_csv) if args.metadata_csv else _find_mimic_csv(
        args.mimic_root, "mimic-cxr-2.0.0-metadata.csv.gz"
    )
    split_path = Path(args.split_csv) if args.split_csv else _find_mimic_csv(
        args.mimic_root, "mimic-cxr-2.0.0-split.csv.gz"
    )
    metadata = pd.read_csv(metadata_path, dtype=str)
    split = pd.read_csv(split_path, dtype=str)
    metadata_index = build_metadata_index(metadata, split)

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("install the 'datasets' dependency before preparing SRRG") from error
    dataset = load_dataset(
        args.dataset_id,
        revision=args.dataset_revision,
        cache_dir=args.cache_dir,
        token=True,
    )
    rows = (row for dataset_split in dataset.values() for row in dataset_split)
    records, stats = build_manifest_records(
        rows,
        metadata_index,
        report_field=args.report_field,
        report_mode=args.report_mode,
    )
    if not records:
        raise RuntimeError("no PA/AP SRRG records matched the MIMIC metadata")
    write_manifests(records, args.manifest_dir, overwrite=args.overwrite)
    counts = {name: sum(record["split"] == name for record in records) for name in ("train", "valid")}
    print(json.dumps({"manifest_dir": str(Path(args.manifest_dir)), "counts": counts, "stats": stats}, indent=2))


if __name__ == "__main__":
    main()

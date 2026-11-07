import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from final.data import FinAlDataset, multimodal_collate_fn, read_manifest, resolve_image_path
from final.transforms import DataTransforms
from prepare_srrg import (
    build_manifest_records,
    build_metadata_index,
    extract_report,
    extract_srrg_leaves,
    write_manifests,
)


class FakeTokenizer:
    def __call__(self, text, return_tensors, truncation, padding, max_length):
        del text, return_tensors, truncation, padding
        input_ids = torch.zeros((1, max_length), dtype=torch.long)
        input_ids[0, :4] = torch.tensor([101, 10, 11, 102])
        attention_mask = input_ids.ne(0).long()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": torch.zeros_like(input_ids),
        }


class PrepareManifestTest(unittest.TestCase):
    def test_pa_ap_filter_and_original_split_mapping(self):
        metadata = pd.DataFrame(
            [
                {"dicom_id": "d1", "subject_id": "100001", "study_id": "11", "ViewPosition": "PA"},
                {"dicom_id": "d2", "subject_id": "100002", "study_id": "12", "ViewPosition": "AP"},
                {"dicom_id": "d3", "subject_id": "100003", "study_id": "13", "ViewPosition": "LATERAL"},
            ]
        )
        splits = pd.DataFrame(
            [
                {"dicom_id": "d1", "subject_id": "100001", "study_id": "11", "split": "train"},
                {"dicom_id": "d2", "subject_id": "100002", "study_id": "12", "split": "test"},
                {"dicom_id": "d3", "subject_id": "100003", "study_id": "13", "split": "validate"},
            ]
        )
        index = build_metadata_index(metadata, splits)
        first_report = """EXAMINATION: Chest radiograph

FINDINGS:
Lungs and Airways:
- First finding.
- Second finding, unchanged.

Pleura:
- No pleural effusion.

IMPRESSION:
1. First impression.
2. Second impression."""
        second_report = """FINDINGS:
Cardiovascular:
- Stable cardiomediastinal silhouette.

IMPRESSION:
1. No acute cardiopulmonary process."""
        rows = [
            {"structured_report": first_report, "image_paths": ["files/p10/p100001/s11/d1.jpg"]},
            {"structured_report": second_report, "image_paths": ["/prefix/files/p10/p100002/s12/d2.jpg"]},
            {"structured_report": first_report, "image_paths": ["files/p10/p100003/s13/d3.jpg"]},
            {"structured_report": first_report, "image_paths": ["files/p10/p100004/s14/d4.jpg"]},
        ]
        records, stats = build_manifest_records(rows, index)
        self.assertEqual(len(records), 2)
        self.assertEqual([record["split"] for record in records], ["train", "valid"])
        self.assertEqual(
            records[0]["report"],
            "First finding Second finding unchanged No pleural effusion "
            "First impression Second impression",
        )
        self.assertEqual(stats["unmatched"], 2)
        self.assertEqual(set(records[0]), {"image_path", "report", "split"})
        with tempfile.TemporaryDirectory() as directory:
            write_manifests(records, directory)
            train_record = json.loads((Path(directory) / "train.jsonl").read_text())
            valid_record = json.loads((Path(directory) / "valid.jsonl").read_text())
            self.assertEqual(train_record["split"], "train")
            self.assertEqual(valid_record["split"], "valid")
            with self.assertRaises(FileExistsError):
                write_manifests(records, directory)

    def test_huggingface_native_report_mode(self):
        structured = """FINDINGS:
Lungs and Airways:
- No focal opacity.

IMPRESSION:
1. No acute disease."""
        record = {"structured_report": structured}
        self.assertEqual(
            extract_report(record, report_mode="final-leaves"),
            "No focal opacity No acute disease",
        )
        self.assertEqual(
            extract_report(record, report_mode="hf-native"),
            "FINDINGS: Lungs and Airways: - No focal opacity. IMPRESSION: 1. No acute disease.",
        )
        with self.assertRaisesRegex(ValueError, "requires --report-field structured_report"):
            extract_report(
                {"findings_section": "No focal opacity."},
                report_field="findings_section",
                report_mode="final-leaves",
            )

    def test_structured_leaf_extraction_rejects_unstructured_text(self):
        self.assertEqual(extract_srrg_leaves("plain report text"), [])
        self.assertEqual(extract_srrg_leaves(None), [])


class DatasetTest(unittest.TestCase):
    def test_manifest_path_resolution_and_collation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "mimic" / "2.1.0" / "files" / "p10"
            image_dir.mkdir(parents=True)
            image_path = image_dir / "one.jpg"
            Image.new("L", (48, 64), color=128).save(image_path)
            manifest_dir = root / "manifests"
            manifest_dir.mkdir()
            record = {
                "image_path": "2.1.0/files/p10/one.jpg",
                "report": "No focal airspace opacity.",
                "split": "train",
            }
            (manifest_dir / "train.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

            resolved = resolve_image_path(root / "mimic", record["image_path"])
            self.assertEqual(resolved, image_path)
            with self.assertRaises(FileNotFoundError):
                resolve_image_path(root / "mimic", "2.1.0/files/missing.jpg")
            dataset = FinAlDataset(
                mimic_root=root / "mimic",
                manifest_dir=manifest_dir,
                split="train",
                text_encoder="unused",
                transform=DataTransforms(is_train=False, crop_size=224),
                image_size=256,
                max_words=8,
                tokenizer=FakeTokenizer(),
            )
            sample = dataset[0]
            self.assertEqual(tuple(sample[0].shape), (3, 224, 224))
            batch = multimodal_collate_fn([sample, sample])
            self.assertEqual(tuple(batch["imgs"].shape), (2, 3, 224, 224))
            self.assertEqual(tuple(batch["caption_ids"].shape), (2, 8))

    def test_manifest_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            path.write_text('{"image_path": "x"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing fields"):
                read_manifest(directory, "train")

            path.write_text(
                json.dumps({"image_path": "x", "report": [], "split": "train"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "non-empty string"):
                read_manifest(directory, "train")


if __name__ == "__main__":
    unittest.main()

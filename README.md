# FinAl

Official repository of "FinAl: Fine-grained Alignment for Detail-Preserving Medical Vision-Language Pretraining (NeurIPS, 2026)"

## Installation

```bash
conda env create -f environment.yml
conda activate final
```

Run all commands from the repository root. SRR-BERT-Leaves weights, M3AE weights, and datasets are not included.

Store MIMIC-CXR-JPG and the M3AE checkpoint in any local directory, then provide their paths using `--mimic-root` and `--vision-checkpoint`. Refer to the CARZero repository for the M3AE checkpoint. SRR-BERT-Leaves is downloaded automatically from Hugging Face, or a local path can be provided using `--text-encoder`.

## Data preparation

Access to [`StanfordAIMI/srrg_findings`](https://huggingface.co/datasets/StanfordAIMI/srrg_findings) and MIMIC-CXR-JPG is required.

```bash
huggingface-cli login

python prepare_srrg.py \
  --mimic-root /path/to/mimic-cxr-jpg \
  --manifest-dir /path/to/manifests \
  --cache-dir /path/to/huggingface-cache
```

This creates `train.jsonl` and `valid.jsonl`, which index the PA/AP image paths and reports used for training and validation.

## Training

The reference configuration uses SRR-BERT-Leaves and a local M3AE vision checkpoint.

```bash
python train.py \
  --mimic-root /path/to/mimic-cxr-jpg \
  --manifest-dir /path/to/manifests \
  --output-dir /path/to/output \
  --text-encoder StanfordAIMI/SRR-BERT-Leaves \
  --vision-checkpoint /path/to/VITB-16-M3AE_last.ckpt \
  --devices 2 \
  --strategy ddp \
  --max-epochs 200 \
  --batch-size 256 \
  --num-workers 24
```

## Inference

Create a UTF-8 text file containing one prompt per line, then run:

```bash
python infer.py \
  --checkpoint /path/to/final.ckpt \
  --image /path/to/image.jpg \
  --prompts-file /path/to/prompts.txt \
  --device cuda:0
```

The result is printed as JSON with cosine similarities and prompt rankings.

Use `python train.py --help`, `python prepare_srrg.py --help`, or `python infer.py --help` for all options.

## Acknowledgement

The FinAl implementation was developed with substantial reference to the [MGCA](https://github.com/HKU-MedAI/MGCA) codebase.

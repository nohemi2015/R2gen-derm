# R2Gen-derm: dermatological report generation across skin tones

Code for the paper **"Generation of dermatological reports from images of skin lesions using memory-based Transformer"** (N. Sánchez-Medel, R. Díaz-Hernández, K. A. Ramírez-Gutiérrez, S. Zapotecas-Martínez, L. Altamirano-Robles, J. J. Velázquez-Arreola; INAOE, Mexico), submitted to *Biomedical Signal Processing and Control*.

This repository adapts **R2Gen** (Chen et al., EMNLP 2020, <https://github.com/zhjohnchan/R2Gen>) to dermoscopic images. The original R2Gen README is kept in [`README_R2Gen.md`](README_R2Gen.md); please cite the original work as well (see below).

## What was changed with respect to R2Gen

| Change | Files |
| --- | --- |
| ViT-L/16 and ViT-B/16 (timm, ImageNet-21k) visual extractors; `d_vf` set automatically (ViT-L 1024, ViT-B 768, ResNet-101 2048) | `modules/visual_extractor.py`, `main_*.py` |
| Extractor-specific image normalisation | `modules/dataloaders.py` |
| Ablation variants `--variant base / rm / rm_mcln` (BASE, +RM, +RM+MCLN) | `modules/encoder_decoder.py` |
| Vocabulary saved at training (`vocab.json`) and reused at test time | `modules/tokenizers.py` |
| `predictions.csv` and `metrics.json` written for every test run | `modules/tester.py` |
| Only the validation-selected checkpoint is reported | `modules/trainer.py` |
| CPU support and pandas 2 compatibility | `modules/*.py` |

New tools:

- `tools/make_splits.py` – leakage audit and lesion-level 70/10/20 splits.
- `tools/build_brown_skin.py` – lesion-level HAM10000 splits, Brown Skin training images and light/dark test pairs.
- `tools/evaluate.py` – BLEU-1–4, METEOR, ROUGE-L (COCO toolkit), diagnosis accuracy and macro-F1 extracted from the generated text; per class, per phototype and per tone; mean ± s.d. over seeds.
- `tools/paired_tone.py` – paired light vs dark comparison with exact McNemar test.
- `tools/baselines.py` – nearest-neighbour retrieval and classifier + template baselines.
- `tools/tone_controls.py` – blur-matched and colour-matched (Reinhard, CIELAB) versions of the light images, to separate skin tone from loss of detail.
- `tools/collect_tables.py` – result tables.
- `tools/finetune_classifier.py` – strong baseline: fully fine-tuned ViT-L with three heads (diagnosis, confirmation type, localisation) + template.
- `modules/tone_aug.py`, `tools/build_tone_refs.py` – skin-tone augmentation balanced across classes (Reinhard transfer to the colour of Brown Skin training images only).

## Requirements

Python 3.10+, PyTorch with CUDA, `timm`, `pandas`, `scikit-learn`, `scipy`, `Pillow`, and Java (for METEOR):

```
pip install -r requirements.txt
```

## Data

- **HAM10000** (Tschandl et al., 2018) and the **ISIC 2018 Task 3 test set** (Codella et al., 2019): <https://challenge.isic-archive.com/data/>
- **IU X-Ray** (Demner-Fushman et al., 2016), with the R2Gen annotation file.
- **Brown Skin** (Sánchez-Medel et al.): dermoscopic images re-rendered on Fitzpatrick phototype VI by neural style transfer. This work uses a random subset of 182 images.

Expected layout (relative to this folder):

```
../data/ham10000/images/{HAM10000_images_all, ISIC2018_Task3_Test_Images, "100 malignas", "100 benignas"}
../data/ham10000/annotation_ham10000_full.json
../data/ham10000/HAM10000_metadata.csv
../data/ham10000/ISIC2018_Task3_Test_GroundTruth.csv
../data/iu_xray/{images, annotation.json}
```

## Reproducing the experiments

```
# 1. Annotation files (lesion-level splits + light/dark pairs)
python tools/build_brown_skin.py --ham_ann ../data/ham10000/annotation_ham10000_full.json \
    --ham_metadata ../data/ham10000/HAM10000_metadata.csv \
    --isic_gt ../data/ham10000/ISIC2018_Task3_Test_GroundTruth.csv \
    --image_root ../data/ham10000/images --bs_dirs "100 malignas" "100 benignas" \
    --out_dir ../data/brown_skin_v2

# 2. Main experiments (3 seeds each; add --dry_run to print the commands)
python scripts/run_experiments.py ham10000    --image_dir ../data/ham10000/images --ann_dir ../data/brown_skin_v2
python scripts/run_experiments.py ham10000_bs --image_dir ../data/ham10000/images --ann_dir ../data/brown_skin_v2
python scripts/run_experiments.py iu_xray     --image_dir ../data/iu_xray/images  --iu_ann ../data/iu_xray/annotation.json

# 3. Skin tone vs loss of detail (no retraining)
python tools/tone_controls.py --ann_path ../data/brown_skin_v2/bs_paired_test.json \
    --image_root ../data/ham10000/images --out_ann ../data/brown_skin_v2/bs_controls_test.json
python scripts/run_tone_controls.py ham10000 --image_dir ../data/ham10000/images \
    --ann ../data/brown_skin_v2/bs_controls_test.json
```

| Experiment | Training data | Evaluation |
| --- | --- | --- |
| `ham10000` | HAM10000 (lesion-level split) | ISIC 2018 test set (1,511) and 83 light/dark pairs |
| `ham10000_bs` | HAM10000 + 89 Brown Skin images | same |
| `iu_xray` | IU X-Ray | IU X-Ray |
| `ham10000_clf` | HAM10000, fine-tuned classifier + template | same as `ham10000` |
| `ham10000_toneaug` | HAM10000 with class-balanced skin-tone augmentation | same as `ham10000` |

Configurations in `ham10000`: ResNet-101 and ViT-L/16 with BASE, +RM and +RM+MCLN, and ViT-B/16 with +RM+MCLN, plus the two baselines. Tables are written to `results/<experiment>/tables`, `tables_bs_pairs` and `tables_tone_controls`.

## Citation

If you use this code, please cite our paper (reference to be added on publication) and R2Gen:

```
@inproceedings{chen-emnlp-2020-r2gen,
    title = "Generating Radiology Reports via Memory-driven Transformer",
    author = "Chen, Zhihong and Song, Yan and Chang, Tsung-Hui and Wan, Xiang",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing",
    year = "2020",
}
```

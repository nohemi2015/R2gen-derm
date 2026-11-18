# Results reported in the paper

Per-image predictions (`*_runN_detail.csv`: id, reference, generated report, true and stated diagnosis) and summary tables for every model and seed, as produced by `tools/evaluate.py`, `tools/paired_tone.py` and `tools/revision_stats.py`.

| Folder | Content |
| --- | --- |
| `ham10000_desc_tables`, `ham10000_desc_bs_pairs` | Descriptive reports: R2Gen, ViT-L BASE, proposed; ISIC 2018 test and Brown Skin pairs |
| `ham10000_clf_desc_tables` | Fine-tuned classifier predictions written as descriptive reports |
| `ham10000_short_tables`, `ham10000_short_bs_pairs` | Short reports: all ablations and baselines |
| `ham10000_clf_tables`, `ham10000_clf_bs_pairs` | Fine-tuned classifier with short reports |
| `ham10000_bs_short_*` | Exploratory experiment with Brown Skin images added to training |
| `iu_xray_tables` | IU X-Ray |
| `revision_stats` | Lesion-clustered bootstrap intervals, McNemar tables and confusion matrices |

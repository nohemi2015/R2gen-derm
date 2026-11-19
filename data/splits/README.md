# Annotation files and lesion-level splits

| File | Content |
| --- | --- |
| `ham10000_lesion.json` | HAM10000 train/val (split by `lesion_id`) and ISIC 2018 Task 3 test (1,511 of 1,512 images), short reports |
| `ham10000_lesion_desc.json` | Same splits, descriptive reports (`tools/build_descriptive_reports.py`) |
| `ham10000_plus_bs.json` | Exploratory experiment: the same splits plus 89 synthetically darkened Brown Skin images (79 train, 10 val) |
| `bs_paired_test.json`, `bs_paired_test_desc.json` | 83 ISIC 2018 test lesions and their synthetically darkened versions (Brown Skin) |
| `isic2018_test_lesion_groups.csv` | `id,group`: lesion identifier of every test image, used for the lesion-clustered bootstrap |

Image files are not included: HAM10000 and ISIC 2018 are available from their providers; Brown Skin images are available from the corresponding author upon reasonable request.
The ISIC 2018 test image ISIC_0035068 (nevus; empty metadata row in the lesion-metadata file) is not included.
Vocabularies are rebuilt deterministically from the training split (words occurring at least three times).

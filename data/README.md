# Dataset

The dataset is **not included** in this repository due to licensing.

## Download Instructions

### Option 1 — Kaggle (recommended)
1. Go to: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
2. Download `spam.csv`
3. Place it in this `data/` folder

### Option 2 — UCI ML Repository
1. Go to: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
2. Download and extract the zip
3. Rename the file to `spam.csv` and place it here

## Format

The CSV should have at minimum these two columns:

| Column | Values          |
|--------|-----------------|
| `v1`   | `ham` or `spam` |
| `v2`   | Message text    |

## Stats

| Class | Count | Percentage |
|-------|-------|------------|
| ham   | 4,827 | 86.6%      |
| spam  | 747   | 13.4%      |
| **Total** | **5,574** | |

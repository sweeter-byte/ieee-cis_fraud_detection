# Dataset Directory

This directory should contain the IEEE-CIS Fraud Detection dataset files.

## Required Files

Download from [Kaggle IEEE-CIS Competition](https://www.kaggle.com/c/ieee-fraud-detection/data):

- `train_transaction.csv` (~684 MB)
- `train_identity.csv` (~27 MB)

## Download Instructions

1. Create a Kaggle account at https://www.kaggle.com
2. Accept the competition rules at the competition page
3. Download using Kaggle CLI:
   ```bash
   pip install kaggle
   kaggle competitions download -c ieee-fraud-detection
   unzip ieee-fraud-detection.zip -d dataset/
   ```

Or download manually from the competition data page.

## Note

The CSV files are excluded from git tracking due to their large size.

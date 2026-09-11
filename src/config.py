"""Central configuration: paths + split sizes. Edit here, nowhere else."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_DIR = ROOT / "data" / "features"

for d in (RAW_DIR, PROCESSED_DIR, FEATURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---- MIND-small (local dev/iteration, per Q1-Q3 work) ----
MIND_TRAIN_DIR = RAW_DIR / "mind" / "train"
MIND_DEV_DIR = RAW_DIR / "mind" / "dev"

# ---- MIND-large + official test set (needed for actual Codabench submission) ----
MIND_LARGE_TRAIN_DIR = RAW_DIR / "mind_large" / "train"
MIND_LARGE_DEV_DIR = RAW_DIR / "mind_large" / "dev"
MIND_LARGE_TEST_DIR = RAW_DIR / "mind_large" / "test"

# ---- EB-NeRD demo (local dev/iteration, per Q1-Q3 work) ----
EBNERD_DIR = RAW_DIR / "ebnerd" / "demo"

# ---- EB-NeRD large + official test set (needed for actual Codabench submission) ----
EBNERD_LARGE_DIR = RAW_DIR / "ebnerd" / "large"
EBNERD_TEST_DIR = RAW_DIR / "ebnerd" / "testset"

# Temporal split sizes (in days), per dataset -- MIND-small and EB-NeRD demo
# cover very different calendar windows, so don't share one constant.
# Run inspect_dates.py first and adjust these to fit each dataset's range.
SPLIT_DAYS = {
    "mind":   {"test_days": 1, "val_days": 1},   # 6-day range: 4 train / 1 val / 1 test
    "ebnerd": {"test_days": 2, "val_days": 2},   # 13-day range: 9 train / 2 val / 2 test
}

DOWNLOAD_URLS = {
    # small/demo -- local dev
    "mind_train": "https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDsmall_train.zip",
    "mind_dev": "https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDsmall_dev.zip",
    "ebnerd_demo": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_demo.zip",

    # large + official test -- required for actual Codabench submission
    "mind_large_train": "https://mind201910small.blob.core.windows.net/release/MINDlarge_train.zip",
    "mind_large_dev": "https://mind201910small.blob.core.windows.net/release/MINDlarge_dev.zip",
    "mind_large_test": "https://mind201910small.blob.core.windows.net/release/MINDlarge_test.zip",

    "ebnerd_large": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_large.zip",
    "ebnerd_articles_large_only": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/articles_large_only.zip",
    "ebnerd_testset": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_testset.zip",
    "ebnerd_predictions_example": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/predictions_large_random.zip",
}

# ---- Behavioural feature engineering (Assignment 2, Part I Q1) ----
N_RECENT_CLICKS = 10          # how many recent clicks to keep in point-in-time history
RECENCY_HALF_LIFE_DAYS = 7.0  # exponential decay half-life for recency-weighted engagement
COLD_START_MAX_CLICKS = 5     # <= this many prior clicks -> cold-start user (for later slicing)
SESSION_GAP_MINUTES = 30.0    # inactivity gap that starts a new session where no session_id exists (MIND)
"""Step 1: download raw MIND-small + EB-NeRD demo, skipping what's already there.
Also supports large-scale + official test downloads, needed only when generating
an actual Codabench submission (not for local Q1-Q3 development)."""
import zipfile
import urllib.request
from pathlib import Path
from src.config import (
    RAW_DIR, MIND_TRAIN_DIR, MIND_DEV_DIR, EBNERD_DIR,
    MIND_LARGE_TRAIN_DIR, MIND_LARGE_DEV_DIR, MIND_LARGE_TEST_DIR,
    EBNERD_LARGE_DIR, EBNERD_TEST_DIR, DOWNLOAD_URLS,
)


def _download(url: str, dest_zip: Path):
    if dest_zip.exists():
        print(f"  [skip] {dest_zip.name} already downloaded")
        return
    print(f"  [get]  {url}")
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest_zip)


def _unzip(zip_path: Path, out_dir: Path):
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"  [skip] {out_dir} already extracted")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [unzip] {zip_path.name} -> {out_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    _flatten_single_nested_dir(out_dir)


def _flatten_single_nested_dir(out_dir: Path):
    """Some zips (MINDlarge_test.zip, ebnerd_testset.zip) contain a single
    top-level folder with the same name as the zip, doubling the nesting
    (e.g. MINDlarge_test/MINDlarge_test/behaviors.tsv). If out_dir contains
    exactly one subdirectory and nothing else, hoist its contents up one
    level so downstream code can rely on flat, predictable paths."""
    import shutil
    entries = list(out_dir.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        nested = entries[0]
        for item in nested.iterdir():
            shutil.move(str(item), str(out_dir / item.name))
        nested.rmdir()
        print(f"  [flatten] removed redundant nested folder {nested.name}/")


def download_mind():
    print("Downloading MIND-small...")
    train_zip = RAW_DIR / "mind" / "MINDsmall_train.zip"
    dev_zip = RAW_DIR / "mind" / "MINDsmall_dev.zip"
    _download(DOWNLOAD_URLS["mind_train"], train_zip)
    _download(DOWNLOAD_URLS["mind_dev"], dev_zip)
    _unzip(train_zip, MIND_TRAIN_DIR)
    _unzip(dev_zip, MIND_DEV_DIR)


def download_ebnerd():
    print("Downloading EB-NeRD demo...")
    demo_zip = RAW_DIR / "ebnerd" / "ebnerd_demo.zip"
    _download(DOWNLOAD_URLS["ebnerd_demo"], demo_zip)
    _unzip(demo_zip, EBNERD_DIR)


def download_mind_large():
    print("Downloading MIND-large (train/dev/test) -- needed for actual Codabench submission...")
    train_zip = RAW_DIR / "mind_large" / "MINDlarge_train.zip"
    dev_zip = RAW_DIR / "mind_large" / "MINDlarge_dev.zip"
    test_zip = RAW_DIR / "mind_large" / "MINDlarge_test.zip"
    _download(DOWNLOAD_URLS["mind_large_train"], train_zip)
    _download(DOWNLOAD_URLS["mind_large_dev"], dev_zip)
    _download(DOWNLOAD_URLS["mind_large_test"], test_zip)
    _unzip(train_zip, MIND_LARGE_TRAIN_DIR)
    _unzip(dev_zip, MIND_LARGE_DEV_DIR)
    _unzip(test_zip, MIND_LARGE_TEST_DIR)


def download_ebnerd_large():
    print("Downloading EB-NeRD large + official test set -- needed for actual Codabench submission...")
    large_zip = RAW_DIR / "ebnerd" / "ebnerd_large.zip"
    test_zip = RAW_DIR / "ebnerd" / "ebnerd_testset.zip"
    _download(DOWNLOAD_URLS["ebnerd_large"], large_zip)
    _download(DOWNLOAD_URLS["ebnerd_testset"], test_zip)
    _unzip(large_zip, EBNERD_LARGE_DIR)
    _unzip(test_zip, EBNERD_TEST_DIR)


def download_ebnerd_example_submission(out_dir: Path = None):
    """Fetch + unzip the organizers' example (random) submission -- inspect this
    to learn the EXACT expected submission file structure for EB-NeRD, no guessing."""
    out_dir = out_dir or (RAW_DIR / "ebnerd" / "example_submission")
    zip_path = RAW_DIR / "ebnerd" / "predictions_large_random.zip"
    _download(DOWNLOAD_URLS["ebnerd_predictions_example"], zip_path)
    _unzip(zip_path, out_dir)
    print(f"  inspect the extracted contents at: {out_dir}")
    return out_dir


def run(datasets, scale: str = "small"):
    """scale: 'small' (default, local dev) or 'large' (Codabench submission)."""
    if scale == "small":
        if "mind" in datasets:
            download_mind()
        if "ebnerd" in datasets:
            download_ebnerd()
    elif scale == "large":
        if "mind" in datasets:
            download_mind_large()
        if "ebnerd" in datasets:
            download_ebnerd_large()
    else:
        raise ValueError(f"unknown scale: {scale}")
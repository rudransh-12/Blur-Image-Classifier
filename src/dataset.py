"""
Dataset manifest generator, group-based splitter, and chunked feature extraction pipeline.
Ensures zero data leakage across scene groups and enforces memory-safe streaming.
"""

import os
import glob
import time
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import psutil
except ImportError:
    psutil = None

from src.config import (
    RAW_DATA_DIR,
    MANIFEST_DIR,
    FEATURE_DIR,
    LOG_DIR,
    FAILED_LOG_PATH,
    FEATURE_NAMES,
    LABEL_MAPPING,
    CLASS_MAPPING,
    BATCH_SIZE,
    RANDOM_STATE,
    FFT_SIZE
)
from src.feature_extractor import extract_features_from_path


def parse_scene_id(filename: str) -> str:
    """
    Extracts the unique scene/group ID from the filename.
    Example: '0_IPHONE-SE_S.JPG' -> '0_IPHONE-SE'
             '102_NIKON-D3400-35MM_F.JPG' -> '102_NIKON-D3400-35MM'
    """
    base_name = os.path.splitext(filename)[0]
    if '_' in base_name:
        # Split off the trailing suffix (_S, _F, _M)
        parts = base_name.rsplit('_', 1)
        if len(parts[1]) <= 2 and parts[1].upper() in ('S', 'F', 'M', 'SHARP', 'DEFOCUS', 'MOTION'):
            return parts[0]
    return base_name


def scan_dataset(dataset_dir: str = RAW_DATA_DIR) -> pd.DataFrame:
    """
    Scans the raw dataset directory and builds a raw manifest DataFrame.
    Does NOT load image pixels into memory.
    """
    records = []
    
    # Supported subfolder mappings in Kwentar dataset
    subdirs = {
        "sharp": "sharp",
        "defocused_blurred": "defocus",
        "defocus": "defocus",
        "motion_blurred": "motion",
        "motion": "motion"
    }

    found_dirs = False
    for folder_name, norm_label in subdirs.items():
        folder_path = os.path.join(dataset_dir, folder_name)
        if os.path.exists(folder_path):
            found_dirs = True
            for file_name in os.listdir(folder_path):
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                    full_path = os.path.abspath(os.path.join(folder_path, file_name))
                    scene_id = parse_scene_id(file_name)
                    binary_label = LABEL_MAPPING.get(norm_label, 0)
                    records.append({
                        "image_path": full_path,
                        "filename": file_name,
                        "scene_id": scene_id,
                        "label": norm_label,
                        "binary_label": binary_label
                    })

    if not found_dirs or len(records) == 0:
        raise FileNotFoundError(f"No valid image files found in dataset directory: {dataset_dir}")

    df = pd.DataFrame(records)
    return df


def verify_no_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Explicitly checks that no scene/group ID exists in more than one split.
    Raises ValueError if data leakage is detected.
    """
    train_scenes = set(train_df['scene_id'].unique())
    val_scenes = set(val_df['scene_id'].unique())
    test_scenes = set(test_df['scene_df'] if 'scene_df' in test_df else test_df['scene_id'].unique())

    train_val_overlap = train_scenes.intersection(val_scenes)
    train_test_overlap = train_scenes.intersection(test_scenes)
    val_test_overlap = val_scenes.intersection(test_scenes)

    if train_val_overlap:
        raise ValueError(f"DATA LEAKAGE ERROR: {len(train_val_overlap)} scene IDs shared between Train and Validation: {list(train_val_overlap)[:5]}")

    if train_test_overlap:
        raise ValueError(f"DATA LEAKAGE ERROR: {len(train_test_overlap)} scene IDs shared between Train and Test: {list(train_test_overlap)[:5]}")

    if val_test_overlap:
        raise ValueError(f"DATA LEAKAGE ERROR: {len(val_test_overlap)} scene IDs shared between Validation and Test: {list(val_test_overlap)[:5]}")

    print("[Validation PASS] Zero scene leakage verified across Train, Validation, and Test splits.")


def print_dataset_statistics(df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Prints comprehensive dataset composition and split statistics.
    """
    print("=" * 60)
    print("DATASET VALIDATION & COMPOSITION")
    print("=" * 60)
    
    # Category counts
    sharp_count = len(df[df['label'] == 'sharp'])
    defocus_count = len(df[df['label'] == 'defocus'])
    motion_count = len(df[df['label'] == 'motion'])
    not_blurry_count = len(df[df['binary_label'] == 1])
    blurry_count = len(df[df['binary_label'] == 0])

    print(f"  Sharp images:          {sharp_count}")
    print(f"  Defocus blurred:       {defocus_count}")
    print(f"  Motion blurred:        {motion_count}")
    print("-" * 60)
    print(f"  NOT_BLURRY (Label 1):  {not_blurry_count}")
    print(f"  BLURRY     (Label 0):  {blurry_count}")
    print(f"  Total Images:          {len(df)}")
    print(f"  Total Unique Scenes:   {df['scene_id'].nunique()}")
    print("-" * 60)
    print("SPLIT BREAKDOWN (Group-Based by Scene ID):")
    print(f"  Train:      {len(train_df)} images ({train_df['scene_id'].nunique()} scenes)")
    print(f"  Validation: {len(val_df)} images ({val_df['scene_id'].nunique()} scenes)")
    print(f"  Test:       {len(test_df)} images ({test_df['scene_id'].nunique()} scenes)")
    print("=" * 60)


def create_group_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset by unique scene IDs so all variations of each scene belong
    to strictly one split.
    """
    unique_scenes = np.array(sorted(df['scene_id'].unique()))
    
    # Deterministic shuffle
    rng = np.random.RandomState(random_state)
    shuffled_scenes = rng.permutation(unique_scenes)

    n_scenes = len(shuffled_scenes)
    n_train = int(n_scenes * train_ratio)
    n_val = int(n_scenes * val_ratio)

    train_scenes = set(shuffled_scenes[:n_train])
    val_scenes = set(shuffled_scenes[n_train:n_train + n_val])
    test_scenes = set(shuffled_scenes[n_train + n_val:])

    train_df = df[df['scene_id'].isin(train_scenes)].copy().reset_index(drop=True)
    val_df = df[df['scene_id'].isin(val_scenes)].copy().reset_index(drop=True)
    test_df = df[df['scene_id'].isin(test_scenes)].copy().reset_index(drop=True)

    # Verify zero leakage
    verify_no_leakage(train_df, val_df, test_df)

    return train_df, val_df, test_df


def save_manifests(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, manifest_dir: str = MANIFEST_DIR) -> Dict[str, str]:
    """
    Saves the lightweight train, validation, and test manifest CSVs.
    Manifest files contain ONLY metadata (no image pixels).
    """
    os.makedirs(manifest_dir, exist_ok=True)
    
    cols = ["image_path", "scene_id", "label", "binary_label"]
    paths = {
        "train": os.path.join(manifest_dir, "train.csv"),
        "validation": os.path.join(manifest_dir, "validation.csv"),
        "test": os.path.join(manifest_dir, "test.csv")
    }

    train_df[cols].to_csv(paths["train"], index=False)
    val_df[cols].to_csv(paths["validation"], index=False)
    test_df[cols].to_csv(paths["test"], index=False)

    print(f"Manifests saved to: {manifest_dir}")
    return paths


def get_memory_usage_mb() -> float:
    """Returns current process RAM usage in MB if psutil is available."""
    if psutil:
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0
    return 0.0


from concurrent.futures import ThreadPoolExecutor
from src.config import N_JOBS

def _extract_single_row(row_tuple, fft_size):
    idx, row = row_tuple
    img_path = row["image_path"]
    binary_lbl = row["binary_label"]
    original_lbl = row.get("label", "unknown")
    scene_id = row.get("scene_id", "unknown")

    try:
        feat_vals = extract_features_from_path(img_path, fft_size=fft_size, as_dict=False)
        record = dict(zip(FEATURE_NAMES, feat_vals))
        record["binary_label"] = binary_lbl
        record["label"] = original_lbl
        record["scene_id"] = scene_id
        record["image_path"] = img_path
        return record, None
    except Exception as e:
        return None, (img_path, str(e))


def extract_features_for_manifest(
    manifest_path: str,
    split_name: str,
    output_feature_dir: str = FEATURE_DIR,
    batch_size: int = BATCH_SIZE,
    fft_size: int = FFT_SIZE,
    n_jobs: int = N_JOBS,
    resume: bool = True
) -> pd.DataFrame:
    """
    Extracts features for a manifest in memory-safe chunks of `batch_size`.
    Saves incremental batch CSV files on disk to support checkpoint/resume.
    Returns the consolidated feature DataFrame.
    """
    df = pd.read_csv(manifest_path)
    total_images = len(df)
    
    split_feature_dir = os.path.join(output_feature_dir, split_name)
    os.makedirs(split_feature_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    n_batches = (total_images + batch_size - 1) // batch_size
    print(f"\n[Feature Extraction: {split_name.upper()}] Total Images: {total_images}, Batch Size: {batch_size}, Batches: {n_batches}, Workers: {n_jobs}")

    successful_batches = []
    total_failed = 0
    start_time = time.time()

    for batch_idx in range(n_batches):
        batch_file = os.path.join(split_feature_dir, f"batch_{batch_idx:04d}.csv")
        
        # Checkpoint / Resume check
        if resume and os.path.exists(batch_file):
            try:
                batch_df = pd.read_csv(batch_file)
                if len(batch_df) > 0 and all(col in batch_df.columns for col in FEATURE_NAMES + ["binary_label"]):
                    successful_batches.append(batch_df)
                    processed_count = min((batch_idx + 1) * batch_size, total_images)
                    pct = (processed_count / total_images) * 100
                    print(f"\r  [Cached Batch {batch_idx + 1:2d}/{n_batches:2d}] {pct:5.1f}% ({processed_count}/{total_images} images)", end="", flush=True)
                    continue
            except Exception:
                pass  # Recompute if file was corrupted

        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_images)
        batch_slice = df.iloc[start_idx:end_idx]

        # Extract features within batch using thread pool (releases memory immediately per image)
        row_items = list(batch_slice.iterrows())
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(lambda item: _extract_single_row(item, fft_size), row_items))

        batch_records = []
        for record, error in results:
            if record is not None:
                batch_records.append(record)
            elif error is not None:
                total_failed += 1
                img_path, err_msg = error
                with open(FAILED_LOG_PATH, "a", encoding="utf-8") as f_log:
                    f_log.write(f"{img_path}\t{err_msg}\n")

        if batch_records:
            batch_df = pd.DataFrame(batch_records)
            # Save batch checkpoint to disk
            batch_df.to_csv(batch_file, index=False)
            successful_batches.append(batch_df)

        # Progress reporting (ASCII-only for cross-platform Windows compatibility)
        processed_count = end_idx
        elapsed = time.time() - start_time
        pct = (processed_count / total_images) * 100
        mem_mb = get_memory_usage_mb()
        
        bar_len = 20
        filled = int(bar_len * processed_count / total_images)
        bar = "=" * filled + "-" * (bar_len - filled)

        mem_str = f" | RAM: {mem_mb:.1f} MB" if mem_mb > 0 else ""
        print(f"\r  [{bar}] {pct:5.1f}% | Batch: {batch_idx + 1:3d}/{n_batches:3d} | Images: {processed_count:4d}/{total_images:4d} | Elapsed: {elapsed:5.1f}s{mem_str}", end="", flush=True)

    print() # New line after progress bar

    # Consolidate all batches
    if successful_batches:
        consolidated_df = pd.concat(successful_batches, ignore_index=True)
    else:
        consolidated_df = pd.DataFrame(columns=FEATURE_NAMES + ["binary_label", "label", "scene_id", "image_path"])

    # Save consolidated split feature file
    consolidated_path = os.path.join(output_feature_dir, f"{split_name}_features.csv")
    consolidated_df.to_csv(consolidated_path, index=False)

    print(f"  -> Finished {split_name.upper()}: {len(consolidated_df)} images processed, {total_failed} failed. Saved to: {consolidated_path}")
    return consolidated_df


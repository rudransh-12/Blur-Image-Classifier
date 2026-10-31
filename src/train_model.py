"""
Model training and evaluation pipeline for lightweight Image Blur Classifier.
Trains a RAM-safe RandomForestClassifier on extracted compact feature matrices.
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from src.config import (
    RAW_DATA_DIR,
    MANIFEST_DIR,
    FEATURE_DIR,
    MODEL_DIR,
    MODEL_PATH,
    FEATURE_NAMES,
    CLASS_MAPPING,
    RF_PARAMS,
    BATCH_SIZE,
    FFT_SIZE
)
from src.dataset import (
    scan_dataset,
    create_group_splits,
    save_manifests,
    print_dataset_statistics,
    verify_no_leakage,
    extract_features_for_manifest,
    get_memory_usage_mb
)


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Dict[str, Any] = RF_PARAMS
) -> RandomForestClassifier:
    """
    Trains a balanced RandomForestClassifier on the compact feature matrix.
    """
    print("\nTraining RandomForestClassifier...")
    print(f"  Hyperparameters: {params}")
    model = RandomForestClassifier(**params)
    model.fit(X_train[FEATURE_NAMES], y_train)
    return model


def train_sgd_fallback(
    train_feature_path: str,
    batch_size: int = 128
) -> SGDClassifier:
    """
    Incremental training fallback using SGDClassifier with log_loss (logistic regression).
    Streams feature chunks to minimize RAM footprint.
    """
    print("\nTraining fallback SGDClassifier (incremental partial_fit)...")
    model = SGDClassifier(loss="log_loss", random_state=42, class_weight="balanced")
    
    classes = np.array([0, 1])
    for chunk in pd.read_csv(train_feature_path, chunksize=batch_size):
        X_chunk = chunk[FEATURE_NAMES]
        y_chunk = chunk["binary_label"]
        model.partial_fit(X_chunk, y_chunk, classes=classes)
        
    return model


def evaluate_model(
    model: Any,
    df_eval: pd.DataFrame,
    split_name: str = "Test"
) -> Dict[str, Any]:
    """
    Evaluates model performance and prints standard metrics along with
    fine-grained blur category recalls (Motion, Defocus, Sharp).
    """
    X = df_eval[FEATURE_NAMES]
    y_true = df_eval["binary_label"]

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else y_pred

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_proba)
    except Exception:
        auc = 0.0

    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 60)
    print(f"MODEL PERFORMANCE ON {split_name.upper()} SET")
    print("=" * 60)
    print(f"  Accuracy:         {acc * 100:.2f}%")
    print(f"  Precision:        {prec * 100:.2f}%")
    print(f"  Recall (Sharp):   {rec * 100:.2f}%")
    print(f"  F1 Score:         {f1 * 100:.2f}%")
    print(f"  ROC-AUC:          {auc:.4f}")
    print("-" * 60)
    print("  Confusion Matrix:")
    print("               Pred BLURRY (0)   Pred NOT_BLURRY (1)")
    print(f"  True BLURRY (0)     {cm[0][0]:<17} {cm[0][1]:<17}")
    if len(cm) > 1:
        print(f"  True NOT_BLURRY (1) {cm[1][0]:<17} {cm[1][1]:<17}")
    print("-" * 60)

    # Subcategory Recalls if original label column is present
    metrics_summary = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm.tolist()
    }

    if "label" in df_eval.columns:
        print("FINE-GRAINED SUB-CATEGORY PERFORMANCE:")
        for category in ["sharp", "defocus", "motion"]:
            cat_mask = df_eval["label"] == category
            if cat_mask.sum() > 0:
                cat_true = y_true[cat_mask]
                cat_pred = y_pred[cat_mask]
                
                if category == "sharp":
                    # For sharp images, correct classification is 1 (NOT_BLURRY)
                    correct = (cat_pred == 1).sum()
                else:
                    # For blur categories, correct classification is 0 (BLURRY)
                    correct = (cat_pred == 0).sum()
                
                cat_rec = correct / len(cat_true)
                metrics_summary[f"{category}_recall"] = cat_rec
                print(f"  {category.capitalize():<12} Accuracy/Recall: {cat_rec * 100:.2f}% ({correct}/{len(cat_true)})")

    # Feature Importance
    if hasattr(model, "feature_importances_"):
        print("-" * 60)
        print("FEATURE IMPORTANCES:")
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        for rank, idx in enumerate(sorted_idx, 1):
            print(f"  {rank}. {FEATURE_NAMES[idx]:<26} {importances[idx] * 100:6.2f}%")
        metrics_summary["feature_importances"] = {FEATURE_NAMES[i]: float(importances[i]) for i in sorted_idx}

    print("=" * 60)
    return metrics_summary


def save_model_bundle(model: Any, model_path: str = MODEL_PATH) -> str:
    """
    Saves the trained model inside a clean, lightweight dictionary bundle.
    Contains strictly model weights and metadata; NO image data or DataFrames.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    bundle = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "class_mapping": CLASS_MAPPING,
        "model_type": model.__class__.__name__,
        "model_version": "1.0",
        "description": "Lightweight generic image blur classifier for SealScan integration"
    }

    joblib.dump(bundle, model_path, compress=3)
    
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"\nModel successfully saved to: {model_path}")
    print(f"  Model bundle file size: {file_size_mb:.3f} MB")
    return model_path


def run_pipeline(
    dataset_dir: str = RAW_DATA_DIR,
    batch_size: int = BATCH_SIZE,
    fft_size: int = FFT_SIZE,
    use_sgd_fallback: bool = False
) -> Dict[str, Any]:
    """
    Executes the end-to-end RAM-safe training pipeline:
    1. Scans raw dataset directory
    2. Builds manifest & performs group-based split (zero scene leakage)
    3. Memory-safe chunked feature extraction to disk
    4. Trains lightweight classifier on compact feature table
    5. Evaluates on validation and test sets
    6. Saves production model bundle
    """
    pipeline_start = time.time()
    print("=" * 60)
    print("STARTING IMAGE BLUR CLASSIFIER TRAINING PIPELINE")
    print("=" * 60)
    print(f"Initial Process RAM: {get_memory_usage_mb():.1f} MB")

    # Step 1: Scan and Manifest
    print("\n[Step 1/5] Scanning dataset and generating metadata manifest...")
    df_raw = scan_dataset(dataset_dir)
    train_manifest_df, val_manifest_df, test_manifest_df = create_group_splits(df_raw)
    
    print_dataset_statistics(df_raw, train_manifest_df, val_manifest_df, test_manifest_df)
    manifest_paths = save_manifests(train_manifest_df, val_manifest_df, test_manifest_df)

    # Step 2: Chunked Feature Extraction
    feat_start = time.time()
    print("\n[Step 2/5] Extracting features in memory-safe chunks (Batch Size: {})...".format(batch_size))
    
    train_features_df = extract_features_for_manifest(
        manifest_paths["train"], "train", batch_size=batch_size, fft_size=fft_size
    )
    val_features_df = extract_features_for_manifest(
        manifest_paths["validation"], "validation", batch_size=batch_size, fft_size=fft_size
    )
    test_features_df = extract_features_for_manifest(
        manifest_paths["test"], "test", batch_size=batch_size, fft_size=fft_size
    )
    feat_time = time.time() - feat_start

    # Step 3: Model Training
    train_start = time.time()
    print("\n[Step 3/5] Training model on compact numerical feature matrix...")
    print(f"  Training feature matrix shape: {train_features_df[FEATURE_NAMES].shape}")
    print(f"  Feature matrix memory footprint: ~{train_features_df[FEATURE_NAMES].memory_usage().sum() / 1024:.2f} KB")

    if use_sgd_fallback:
        model = train_sgd_fallback(os.path.join(FEATURE_DIR, "train_features.csv"))
    else:
        model = train_random_forest(train_features_df, train_features_df["binary_label"])
    train_time = time.time() - train_start

    # Step 4: Validation & Testing
    print("\n[Step 4/5] Evaluating model performance...")
    val_metrics = evaluate_model(model, val_features_df, split_name="Validation")
    test_metrics = evaluate_model(model, test_features_df, split_name="Test")

    # Step 5: Save Model Bundle
    print("\n[Step 5/5] Packaging and saving model bundle...")
    saved_path = save_model_bundle(model, MODEL_PATH)

    total_time = time.time() - pipeline_start
    peak_mem = get_memory_usage_mb()

    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 60)
    print(f"  Total Pipeline Time:       {total_time:.2f} seconds")
    print(f"  Feature Extraction Time:   {feat_time:.2f} seconds")
    print(f"  Model Training Time:       {train_time:.2f} seconds")
    print(f"  Peak Process RAM:          {peak_mem:.1f} MB")
    print(f"  Model Bundle Saved To:     {saved_path}")
    print(f"  Test Accuracy:             {test_metrics['accuracy'] * 100:.2f}%")
    print(f"  Test F1 Score:             {test_metrics['f1'] * 100:.2f}%")
    print(f"  Test ROC-AUC:              {test_metrics['roc_auc']:.4f}")
    print("=" * 60)

    return {
        "model": model,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "times": {
            "total_time": total_time,
            "feature_extraction_time": feat_time,
            "train_time": train_time
        }
    }


if __name__ == "__main__":
    run_pipeline()

"""
Standalone Inference Example for SealScan Integration.

Demonstrates loading the blur classifier model bundle and evaluating images
without requiring training pipelines, GPU, or deep learning frameworks.
"""

import sys
import os
import joblib
import pandas as pd

# Add workspace to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.feature_extractor import extract_features_from_path


def run_inference_example(
    image_path: str,
    model_path: str = "models/blur_classifier.pkl"
):
    """
    Standard lightweight inference workflow for SealScan.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model not found at '{model_path}'. Please train the model first by running train_model.py.")
        return

    if not os.path.exists(image_path):
        print(f"Error: Image not found at '{image_path}'.")
        return

    print(f"\nEvaluating Image: {image_path}")
    print("-" * 50)

    # 1. Load the lightweight bundle
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_names = bundle["feature_names"]

    # 2. Extract the 7 sharpness/blur features
    features = extract_features_from_path(image_path)

    # 3. Create single-row input DataFrame
    X = pd.DataFrame([features], columns=feature_names)

    # 4. Predict probabilities
    probabilities = model.predict_proba(X)[0]
    probability_blurry = float(probabilities[0])
    probability_not_blurry = float(probabilities[1])

    prediction = "NOT_BLURRY" if probability_not_blurry >= 0.5 else "BLURRY"

    result = {
        "prediction": prediction,
        "probability_blurry": round(probability_blurry, 4),
        "probability_not_blurry": round(probability_not_blurry, 4)
    }

    print("Result:")
    print(f"  Prediction:              {result['prediction']}")
    print(f"  Probability (BLURRY):     {result['probability_blurry']:.4f}")
    print(f"  Probability (NOT_BLURRY): {result['probability_not_blurry']:.4f}")
    print("\nExtracted Feature Metrics:")
    for name, val in zip(feature_names, features):
        print(f"  - {name:<26}: {val:.6f}")
    print("-" * 50)

    return result


if __name__ == "__main__":
    # Test on default sample images from dataset if present
    sample_images = [
        "blur_dataset/sharp/0_IPHONE-SE_S.JPG",
        "blur_dataset/defocused_blurred/0_IPHONE-SE_F.JPG",
        "blur_dataset/motion_blurred/0_IPHONE-SE_M.JPG"
    ]

    # If an image argument is provided via CLI
    if len(sys.argv) > 1:
        run_inference_example(sys.argv[1])
    else:
        print("Running inference test across sample images...")
        for sample in sample_images:
            if os.path.exists(sample):
                run_inference_example(sample)

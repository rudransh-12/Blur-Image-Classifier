"""
Inference module for Image Blur Classifier.
Provides standalone, lightweight prediction functions directly integrable into SealScan.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Union

from src.feature_extractor import extract_features_from_path, extract_features, FEATURE_NAMES


class BlurClassifier:
    """
    Lightweight Blur Classifier inference wrapper for SealScan.
    """

    def __init__(self, model_path: str = "models/blur_classifier.pkl"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model bundle not found at: {model_path}")

        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.feature_names = bundle.get("feature_names", FEATURE_NAMES)
        self.class_mapping = bundle.get("class_mapping", {0: "BLURRY", 1: "NOT_BLURRY"})
        self.model_version = bundle.get("model_version", "1.0")

    def predict_from_path(self, image_path: str) -> Dict[str, Any]:
        """
        Extracts features from an image on disk and predicts blur probabilities.
        Memory-safe: releases grayscale image buffer immediately after extraction.
        """
        features_list = extract_features_from_path(image_path, as_dict=False)
        return self._predict_from_features(features_list)

    def predict_from_array(self, image_array: np.ndarray) -> Dict[str, Any]:
        """
        Extracts features from an in-memory image array and predicts blur probabilities.
        """
        features_list = extract_features(image_array, as_dict=False)
        return self._predict_from_features(features_list)

    def _predict_from_features(self, features_list: list) -> Dict[str, Any]:
        """
        Internal predictor taking ordered feature values list.
        """
        X = pd.DataFrame([features_list], columns=self.feature_names)
        
        # Binary probabilities: [P(BLURRY=0), P(NOT_BLURRY=1)]
        probabilities = self.model.predict_proba(X)[0]
        prob_blurry = float(probabilities[0])
        prob_not_blurry = float(probabilities[1])

        # Standard discrete classification (threshold 0.5)
        pred_label_idx = int(np.argmax(probabilities))
        prediction = self.class_mapping.get(pred_label_idx, "BLURRY" if pred_label_idx == 0 else "NOT_BLURRY")

        feature_dict = dict(zip(self.feature_names, features_list))

        return {
            "prediction": prediction,
            "probability_blurry": prob_blurry,
            "probability_not_blurry": prob_not_blurry,
            "features": feature_dict
        }


def predict_image(image_path: str, model_path: str = "models/blur_classifier.pkl") -> Dict[str, Any]:
    """
    Convenience function to classify a single image directly.
    """
    classifier = BlurClassifier(model_path=model_path)
    return classifier.predict_from_path(image_path)

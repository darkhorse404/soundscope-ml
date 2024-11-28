import pytest
import numpy as np
from src.models.classifier import SoundClassifier

def test_train_classifier():
    classifier = SoundClassifier()
    X_train = np.random.randn(100, 13)  # 100 samples, 13 features (e.g., MFCCs)
    y_train = np.random.randint(0, 2, size=100)  # Binary classification
    classifier.train(X_train, y_train)
    assert classifier.classifier is not None  # Ensure model is trained

def test_predict():
    classifier = SoundClassifier()
    X_train = np.random.randn(100, 13)
    y_train = np.random.randint(0, 2, size=100)
    classifier.train(X_train, y_train)
    
    X_test = np.random.randn(10, 13)
    predictions = classifier.predict(X_test)
    assert predictions.shape == (10,)  # Check number of predictions matches test samples

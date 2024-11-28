import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from typing import Tuple, List, Optional
import joblib
import logging

logger = logging.getLogger(__name__)

class SoundClassifier:
    """Sound classification model using Random Forest."""
    
    def __init__(self):
        """Initialize the classifier with default parameters."""
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.classes = None
        
    def train(self, X: np.ndarray, y: np.ndarray, 
             optimize: bool = True) -> None:
        """
        Train the classifier.
        
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Training labels
            optimize (bool): Whether to perform hyperparameter optimization
        """
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            if optimize:
                self._optimize_hyperparameters(X_scaled, y)
            else:
                self.classifier.fit(X_scaled, y)
            
            self.classes = self.classifier.classes_
            logger.info("Model training completed successfully")
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predicted labels
        """
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Class probabilities
        """
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict_proba(X_scaled)
    
    def _optimize_hyperparameters(self, X: np.ndarray, 
                                y: np.ndarray) -> None:
        """
        Perform grid search for hyperparameter optimization.
        
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Training labels
        """
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        grid_search = GridSearchCV(
            estimator=self.classifier,
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        self.classifier = grid_search.best_estimator_
        logger.info(f"Best parameters: {grid_search.best_params_}")
    
    def save_model(self, path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path (str): Path to save the model
        """
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'classes': self.classes
        }
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            path (str): Path to the saved model
        """
        model_data = joblib.load(path)
        self.classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        self.classes = model_data['classes']
        logger.info(f"Model loaded from {path}")
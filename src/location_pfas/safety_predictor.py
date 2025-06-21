import pandas as pd
import numpy as np
import pickle
import json
from typing import Dict, List, Tuple, Optional, Union
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

class PFASSafetyPredictor:
    """MLP-based system for predicting water safety based on PFAS concentrations."""
    
    def __init__(self):
        """Initialize the PFAS safety predictor."""
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.feature_names = []
        self.is_trained = False
        self.safety_thresholds = {
            'safe': 10.0,      # ppb
            'low': 20.0,       # ppb
            'moderate': 50.0,  # ppb
            'high': float('inf')  # ppb
        }
        
    def create_training_data(self) -> pd.DataFrame:
        """Create comprehensive training data for PFAS safety prediction.
        
        Returns:
            DataFrame with PFAS concentrations and safety labels
        """
        print("🔬 Creating PFAS safety training dataset...")
        
        # Common PFAS compounds and their typical concentration ranges
        pfas_compounds = {
            'PFOA': {'id': '335-67-1', 'name': 'Perfluorooctanoic acid', 'max_safe': 0.004},
            'PFOS': {'id': '1763-23-1', 'name': 'Perfluorooctane sulfonic acid', 'max_safe': 0.02},
            'PFBS': {'id': '375-73-5', 'name': 'Perfluorobutane sulfonic acid', 'max_safe': 2.0},
            'PFHxS': {'id': '355-46-4', 'name': 'Perfluorohexane sulfonic acid', 'max_safe': 0.02},
            'PFNA': {'id': '375-95-1', 'name': 'Perfluorononanoic acid', 'max_safe': 0.004},
            'PFDA': {'id': '335-76-2', 'name': 'Perfluorodecanoic acid', 'max_safe': 0.004},
            'PFHxA': {'id': '307-24-4', 'name': 'Perfluorohexanoic acid', 'max_safe': 0.004},
            'PFBA': {'id': '375-22-4', 'name': 'Perfluorobutanoic acid', 'max_safe': 0.004}
        }
        
        # Generate synthetic training data
        np.random.seed(42)  # For reproducibility
        n_samples = 10000
        
        data = []
        
        for _ in range(n_samples):
            # Generate random PFAS concentrations
            sample = {}
            
            # Total PFAS concentration (realistic range: 0-200 ppb)
            total_pfas = np.random.exponential(15)  # Exponential distribution for realistic skew
            
            # Individual compound concentrations
            compound_concentrations = {}
            remaining_concentration = total_pfas
            
            for compound_name, compound_info in pfas_compounds.items():
                # Generate concentration based on compound's typical range
                max_conc = compound_info['max_safe'] * 1000  # Convert to ppb
                conc = np.random.exponential(max_conc / 10)
                conc = min(conc, remaining_concentration * 0.3)  # Limit individual contribution
                
                compound_concentrations[compound_name] = conc
                remaining_concentration -= conc
                
                if remaining_concentration <= 0:
                    break
            
            # Fill remaining concentration with other compounds
            if remaining_concentration > 0:
                for compound_name in pfas_compounds.keys():
                    if compound_name not in compound_concentrations:
                        conc = np.random.uniform(0, remaining_concentration * 0.1)
                        compound_concentrations[compound_name] = conc
                        remaining_concentration -= conc
                        if remaining_concentration <= 0:
                            break
            
            # Calculate safety metrics
            safety_score = self._calculate_safety_score(compound_concentrations, pfas_compounds)
            safety_category = self._categorize_safety(safety_score)
            
            # Create sample
            sample_data = {
                'total_pfas_ppb': total_pfas,
                'safety_score': safety_score,
                'safety_category': safety_category
            }
            
            # Add individual compound concentrations
            for compound_name, conc in compound_concentrations.items():
                sample_data[f'{compound_name}_ppb'] = conc
            
            data.append(sample_data)
        
        df = pd.DataFrame(data)
        
        # Add derived features
        df['log_total_pfas'] = np.log1p(df['total_pfas_ppb'])
        df['pfas_diversity'] = (df[[col for col in df.columns if col.endswith('_ppb')]] > 0).sum(axis=1)
        df['max_compound_conc'] = df[[col for col in df.columns if col.endswith('_ppb')]].max(axis=1)
        df['compound_ratio'] = df['max_compound_conc'] / df['total_pfas_ppb']
        
        print(f"✅ Created training dataset with {len(df)} samples")
        return df
    
    def _calculate_safety_score(self, concentrations: Dict[str, float], 
                               compound_info: Dict) -> float:
        """Calculate a safety score based on PFAS concentrations.
        
        Args:
            concentrations: Dictionary of compound concentrations
            compound_info: Dictionary of compound information
            
        Returns:
            Safety score (lower is safer)
        """
        score = 0.0
        
        for compound_name, conc in concentrations.items():
            if compound_name in compound_info:
                max_safe = compound_info[compound_name]['max_safe'] * 1000  # Convert to ppb
                if conc > 0:
                    # Calculate risk based on concentration relative to safe level
                    risk_ratio = conc / max_safe
                    score += risk_ratio
        
        return score
    
    def _categorize_safety(self, safety_score: float) -> str:
        """Categorize safety based on safety score.
        
        Args:
            safety_score: Calculated safety score
            
        Returns:
            Safety category
        """
        if safety_score < 1.0:
            return 'safe'
        elif safety_score < 5.0:
            return 'low_risk'
        elif safety_score < 20.0:
            return 'moderate_risk'
        else:
            return 'high_risk'
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare features for model training.
        
        Args:
            df: Training dataframe
            
        Returns:
            Tuple of (X, y_classification, y_regression)
        """
        # Feature columns
        feature_cols = [col for col in df.columns if col.endswith('_ppb') or 
                       col.startswith('log_') or col.startswith('pfas_') or 
                       col.startswith('max_') or col.startswith('compound_')]
        
        X = df[feature_cols].values
        y_classification = df['safety_category'].values
        y_regression = df['safety_score'].values
        
        self.feature_names = feature_cols
        
        return X, y_classification, y_regression
    
    def train_model(self, X: np.ndarray, y_classification: np.ndarray, 
                   y_regression: np.ndarray) -> None:
        """Train the MLP model for safety prediction.
        
        Args:
            X: Feature matrix
            y_classification: Classification targets
            y_regression: Regression targets
        """
        print("🧠 Training MLP safety prediction model...")
        
        # Split data
        X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
            X, y_classification, y_regression, test_size=0.2, random_state=42, stratify=y_classification
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Encode classification labels
        y_class_train_encoded = self.label_encoder.fit_transform(y_class_train)
        y_class_test_encoded = self.label_encoder.transform(y_class_test)
        
        # Train classification model
        print("   Training classification model...")
        self.classification_model = MLPClassifier(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        self.classification_model.fit(X_train_scaled, y_class_train_encoded)
        
        # Train regression model
        print("   Training regression model...")
        self.regression_model = MLPRegressor(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        self.regression_model.fit(X_train_scaled, y_reg_train)
        
        # Evaluate models
        self._evaluate_models(X_test_scaled, y_class_test_encoded, y_reg_test)
        
        self.is_trained = True
        print("✅ Model training completed!")
    
    def _evaluate_models(self, X_test: np.ndarray, y_class_test: np.ndarray, 
                        y_reg_test: np.ndarray) -> None:
        """Evaluate the trained models.
        
        Args:
            X_test: Test features
            y_class_test: Test classification targets
            y_reg_test: Test regression targets
        """
        print("\n📊 Model Evaluation:")
        print("-" * 40)
        
        # Classification evaluation
        y_class_pred = self.classification_model.predict(X_test)
        class_accuracy = self.classification_model.score(X_test, y_class_test)
        
        print(f"Classification Accuracy: {class_accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_class_test, y_class_pred, 
                                  target_names=self.label_encoder.classes_))
        
        # Regression evaluation
        y_reg_pred = self.regression_model.predict(X_test)
        reg_r2 = r2_score(y_reg_test, y_reg_pred)
        reg_rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
        
        print(f"\nRegression R² Score: {reg_r2:.3f}")
        print(f"Regression RMSE: {reg_rmse:.3f}")
    
    def predict_safety(self, pfas_data: Dict[str, float]) -> Dict[str, Union[str, float, str]]:
        """Predict water safety based on PFAS concentrations.
        
        Args:
            pfas_data: Dictionary with PFAS concentrations
            
        Returns:
            Dictionary with safety prediction results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare features
        features = self._extract_features(pfas_data)
        features_scaled = self.scaler.transform([features])
        
        # Make predictions
        safety_category_encoded = self.classification_model.predict(features_scaled)[0]
        safety_category = self.label_encoder.inverse_transform([safety_category_encoded])[0]
        safety_score = self.regression_model.predict(features_scaled)[0]
        
        # Determine risk level
        risk_level = self._get_risk_level(safety_score)
        
        return {
            'safety_category': safety_category,
            'safety_score': float(safety_score),
            'risk_level': risk_level,
            'recommendation': self._get_recommendation(safety_category, safety_score)
        }
    
    def _extract_features(self, pfas_data: Dict[str, float]) -> np.ndarray:
        """Extract features from PFAS data.
        
        Args:
            pfas_data: Dictionary with PFAS concentrations
            
        Returns:
            Feature array
        """
        features = []
        
        # Individual compound concentrations
        compound_names = ['PFOA', 'PFOS', 'PFBS', 'PFHxS', 'PFNA', 'PFDA', 'PFHxA', 'PFBA']
        for compound in compound_names:
            features.append(pfas_data.get(f'{compound}_ppb', 0.0))
        
        # Total PFAS
        total_pfas = sum(pfas_data.get(f'{compound}_ppb', 0.0) for compound in compound_names)
        features.append(total_pfas)
        
        # Derived features
        features.append(np.log1p(total_pfas))
        features.append(sum(1 for compound in compound_names if pfas_data.get(f'{compound}_ppb', 0.0) > 0))
        features.append(max(pfas_data.get(f'{compound}_ppb', 0.0) for compound in compound_names))
        
        max_compound = max(pfas_data.get(f'{compound}_ppb', 0.0) for compound in compound_names)
        features.append(max_compound / total_pfas if total_pfas > 0 else 0.0)
        
        return np.array(features)
    
    def _get_risk_level(self, safety_score: float) -> str:
        """Get risk level based on safety score.
        
        Args:
            safety_score: Predicted safety score
            
        Returns:
            Risk level string
        """
        if safety_score < 1.0:
            return 'Low Risk'
        elif safety_score < 5.0:
            return 'Moderate Risk'
        elif safety_score < 20.0:
            return 'High Risk'
        else:
            return 'Very High Risk'
    
    def _get_recommendation(self, safety_category: str, safety_score: float) -> str:
        """Get recommendation based on safety prediction.
        
        Args:
            safety_category: Predicted safety category
            safety_score: Predicted safety score
            
        Returns:
            Recommendation string
        """
        if safety_category == 'safe':
            return "Water appears safe for consumption. Continue regular monitoring."
        elif safety_category == 'low_risk':
            return "Low risk detected. Consider additional testing and monitoring."
        elif safety_category == 'moderate_risk':
            return "Moderate risk detected. Implement treatment measures and increase monitoring frequency."
        else:
            return "High risk detected. Immediate action required. Consider alternative water sources."
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model to file.
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'classification_model': self.classification_model,
            'regression_model': self.regression_model,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained,
            'safety_thresholds': self.safety_thresholds
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model from file.
        
        Args:
            filepath: Path to the saved model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.classification_model = model_data['classification_model']
        self.regression_model = model_data['regression_model']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        self.safety_thresholds = model_data['safety_thresholds']
        
        print(f"✅ Model loaded from {filepath}")
    
    def get_model_info(self) -> Dict:
        """Get information about the trained model.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_trained:
            return {'status': 'Not trained'}
        
        return {
            'status': 'Trained',
            'feature_count': len(self.feature_names),
            'classification_classes': list(self.label_encoder.classes_),
            'classification_layers': self.classification_model.hidden_layer_sizes,
            'regression_layers': self.regression_model.hidden_layer_sizes,
            'feature_names': self.feature_names
        } 
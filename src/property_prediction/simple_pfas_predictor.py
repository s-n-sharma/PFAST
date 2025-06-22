"""
Simple PFAS Property Predictor

A minimal, working implementation for PFAS property prediction
using MoleculeNet/DeepChem and RDKit.

Author: PFAS AI Project
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Core imports
import deepchem as dc
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit import DataStructs
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


class SimplePFASPredictor:
    """
    Simple PFAS property predictor using MoleculeNet and RDKit.
    
    This class provides:
    1. Molecular feature extraction using RDKit
    2. Property prediction using pre-trained models or custom models
    3. Basic evaluation and comparison tools
    """
    
    def __init__(self, use_pretrained: bool = True):
        """
        Initialize the predictor.
        
        Args:
            use_pretrained: Whether to use pre-trained MoleculeNet models
        """
        self.use_pretrained = use_pretrained
        self.models = {}
        self.feature_names = []
        
        # Load pre-trained models if requested
        if self.use_pretrained:
            self._load_pretrained_models()
    
    def _load_pretrained_models(self):
        """Load pre-trained MoleculeNet models."""
        try:
            print("Loading pre-trained MoleculeNet models...")
            
            # Load Delaney (ESOL) dataset for solubility prediction
            tasks, datasets, transformers = dc.molnet.load_delaney(
                featurizer='ECFP', 
                split='random'
            )
            train_dataset, valid_dataset, test_dataset = datasets
            
            # Train a simple model on the Delaney dataset
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            # Get features and targets
            X_train = train_dataset.X
            y_train = train_dataset.y.flatten()
            
            # Train the model
            model.fit(X_train, y_train)
            
            self.models['delaney_solubility'] = {
                'model': model,
                'transformer': transformers[0],
                'task_name': 'measured log solubility in mols per litre'
            }
            
            print(f"✓ Loaded Delaney solubility model")
            print(f"  - Training samples: {len(train_dataset)}")
            print(f"  - Validation samples: {len(valid_dataset)}")
            print(f"  - Test samples: {len(test_dataset)}")
            
        except Exception as e:
            print(f"⚠️  Could not load pre-trained models: {e}")
            print("  Will use custom feature-based prediction instead")
    
    def extract_molecular_features(self, smiles_list: List[str]) -> Dict:
        """
        Extract molecular features using RDKit.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Dictionary containing different feature types
        """
        features = {
            'basic': [],
            'rdkit': [],
            'morgan': []
        }
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is None:
                # Add zeros for invalid molecules
                features['basic'].append([0] * 10)
                features['rdkit'].append([0] * 200)
                features['morgan'].append([0] * 2048)
                continue
            
            # Basic features
            basic_feats = [
                Chem.Descriptors.MolWt(mol),
                Chem.Descriptors.MolLogP(mol),
                Chem.Descriptors.TPSA(mol),
                Chem.Descriptors.NumHDonors(mol),
                Chem.Descriptors.NumHAcceptors(mol),
                Chem.Descriptors.NumRotatableBonds(mol),
                Chem.Descriptors.NumAromaticRings(mol),
                Chem.Descriptors.FractionCSP3(mol),
                Chem.Descriptors.HeavyAtomCount(mol),
                Chem.Descriptors.RingCount(mol)
            ]
            features['basic'].append(basic_feats)
            
            # RDKit descriptors
            rdkit_feats = []
            for desc_name, desc_func in Descriptors.descList:
                try:
                    rdkit_feats.append(desc_func(mol))
                except:
                    rdkit_feats.append(0)
            features['rdkit'].append(rdkit_feats[:200])  # Limit to 200 descriptors
            
            # Morgan fingerprints
            morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            morgan_array = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(morgan_fp, morgan_array)
            features['morgan'].append(morgan_array)
        
        # Convert to numpy arrays
        for key in features:
            features[key] = np.array(features[key])
        
        return features
    
    def predict_solubility(self, smiles_list: List[str], method: str = 'auto') -> Dict:
        """
        Predict solubility for PFAS compounds.
        
        Args:
            smiles_list: List of SMILES strings
            method: Prediction method ('auto', 'pretrained', 'custom')
            
        Returns:
            Dictionary containing predictions and metadata
        """
        print(f"Predicting solubility for {len(smiles_list)} compounds...")
        
        # Extract features
        features = self.extract_molecular_features(smiles_list)
        
        predictions = {}
        
        # Method 1: Pre-trained model (if available)
        if method in ['auto', 'pretrained'] and 'delaney_solubility' in self.models:
            try:
                print("Using pre-trained Delaney model...")
                model_info = self.models['delaney_solubility']
                model = model_info['model']
                
                # Use Morgan fingerprints for prediction
                X_pred = features['morgan']
                pred_solubility = model.predict(X_pred)
                
                predictions['delaney_model'] = pred_solubility
                print(f"✓ Pre-trained model predictions completed")
                
            except Exception as e:
                print(f"⚠️  Pre-trained model failed: {e}")
        
        # Method 2: Custom feature-based prediction
        if method in ['auto', 'custom'] or len(predictions) == 0:
            try:
                print("Using custom feature-based prediction...")
                
                # Simple rule-based prediction using LogP and molecular weight
                custom_predictions = []
                
                for i, smiles in enumerate(smiles_list):
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        custom_predictions.append(-3.0)  # Default low solubility
                        continue
                    
                    logp = Chem.Descriptors.MolLogP(mol)
                    mw = Chem.Descriptors.MolWt(mol)
                    tpsa = Chem.Descriptors.TPSA(mol)
                    
                    # Simple solubility prediction based on Lipinski's rules
                    # Lower LogP and higher TPSA generally indicate better solubility
                    predicted_logs = -2.0 - 0.5 * logp + 0.01 * tpsa - 0.001 * mw
                    custom_predictions.append(predicted_logs)
                
                predictions['custom_model'] = np.array(custom_predictions)
                print(f"✓ Custom model predictions completed")
                
            except Exception as e:
                print(f"⚠️  Custom model failed: {e}")
        
        # Combine predictions if multiple methods available
        if len(predictions) > 1:
            # Simple ensemble (average)
            all_preds = np.array(list(predictions.values()))
            ensemble_pred = np.mean(all_preds, axis=0)
            predictions['ensemble'] = ensemble_pred
            print(f"✓ Ensemble predictions completed")
        
        return {
            'smiles': smiles_list,
            'predictions': predictions,
            'features': features
        }
    
    def evaluate_on_test_data(self) -> Dict:
        """Evaluate the model on the test dataset."""
        if 'delaney_solubility' not in self.models:
            print("No pre-trained model available for evaluation")
            return {}
        
        try:
            # Load test data
            tasks, datasets, transformers = dc.molnet.load_delaney(
                featurizer='ECFP', 
                split='random'
            )
            train_dataset, valid_dataset, test_dataset = datasets
            
            model_info = self.models['delaney_solubility']
            model = model_info['model']
            
            # Make predictions on test set
            X_test = test_dataset.X
            y_test = test_dataset.y.flatten()
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'n_samples': len(y_test)
            }
            
            print(f"Test Set Evaluation Results:")
            print(f"  - MSE: {mse:.4f}")
            print(f"  - RMSE: {rmse:.4f}")
            print(f"  - MAE: {mae:.4f}")
            print(f"  - R²: {r2:.4f}")
            print(f"  - Samples: {len(y_test)}")
            
            return results
            
        except Exception as e:
            print(f"Evaluation failed: {e}")
            return {}


def main():
    """Demo the simple PFAS predictor."""
    print("Simple PFAS Property Predictor Demo")
    print("=" * 40)
    
    # Initialize predictor
    predictor = SimplePFASPredictor(use_pretrained=True)
    
    # Test with PFAS compounds
    pfas_compounds = [
        "C(=O)(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O",  # PFOA
        "C(C(C(C(C(F)(F)S(=O)(=O)O)(F)F)(F)F)(F)F)(C(C(C(F)(F)F)(F)F)(F)F)(F)F",  # PFOS
        "C(=O)(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)O",  # PFHxA
        "C(C(C(F)(F)S(=O)(=O)O)(F)F)(C(F)(F)F)(F)F",  # PFBS
    ]
    
    pfas_names = ["PFOA", "PFOS", "PFHxA", "PFBS"]
    
    # Make predictions
    results = predictor.predict_solubility(pfas_compounds)
    
    # Display results
    print(f"\nPrediction Results:")
    print("-" * 30)
    
    for i, (smiles, name) in enumerate(zip(pfas_compounds, pfas_names)):
        print(f"\n{name}:")
        print(f"  SMILES: {smiles}")
        
        for method, preds in results['predictions'].items():
            pred_value = preds[i]
            print(f"  {method}: {pred_value:.3f} logS")
    
    # Evaluate model
    print(f"\nModel Evaluation:")
    print("-" * 20)
    evaluation_results = predictor.evaluate_on_test_data()
    
    print(f"\nDemo completed successfully! 🎉")


if __name__ == "__main__":
    main() 
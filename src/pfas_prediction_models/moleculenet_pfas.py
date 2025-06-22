"""
MoleculeNet PFAS Property Predictor

Uses DeepChem MoleculeNet models for PFAS property prediction.

Author: PFAS AI Project
Date: 2024
"""

import deepchem as dc
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from typing import List, Dict, Optional
import numpy as np
import warnings
import os
import joblib
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score


class MoleculeNetPFASPredictor:
    """
    PFAS property predictor using MoleculeNet/DeepChem models.
    
    Available properties:
    - Solubility (Delaney dataset)
    - Lipophilicity (Lipophilicity dataset) 
    - Blood-brain barrier penetration (BBBP dataset)
    - HIV replication inhibition (HIV dataset)
    """
    
    def __init__(self):
        self.models = {}
        self.property_info = {}
        self.model_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        # Define available properties and their datasets
        self.available_properties = {
            'solubility': {
                'dataset': 'delaney',
                'task_name': 'measured log solubility in mols per litre',
                'type': 'regression',
                'description': 'Aqueous solubility (logS)',
                'unit': 'log(mol/L)',
                'method': 'MoleculeNet Delaney model'
            },
            'lipophilicity': {
                'dataset': 'lipo',
                'task_name': 'exp',
                'type': 'regression', 
                'description': 'Octanol-water partition coefficient',
                'unit': 'logP',
                'method': 'MoleculeNet Lipophilicity model'
            },
            'bbbp': {
                'dataset': 'bbbp',
                'task_name': 'p_np',
                'type': 'classification',
                'description': 'Blood-brain barrier penetration',
                'unit': 'probability',
                'method': 'MoleculeNet BBBP model'
            },
            'hiv': {
                'dataset': 'hiv',
                'task_name': 'HIV_active',
                'type': 'classification',
                'description': 'HIV replication inhibition',
                'unit': 'probability',
                'method': 'MoleculeNet HIV model'
            }
        }
        
        # Load models
        self._load_all_models()
    
    def _get_model_path(self, prop_name):
        return os.path.join(self.model_dir, f"{prop_name}_model.joblib")
    
    def _load_all_models(self):
        """Load all available MoleculeNet models."""
        print("Loading MoleculeNet models for PFAS property prediction...")
        
        for prop_name, prop_info in self.available_properties.items():
            model_path = self._get_model_path(prop_name)
            try:
                if os.path.exists(model_path):
                    # Load model from disk
                    print(f"Loading {prop_name} model from disk...")
                    model = joblib.load(model_path)
                    # Load dataset to get transformers and task_name
                    if prop_info['dataset'] == 'delaney':
                        tasks, datasets, transformers = dc.molnet.load_delaney(
                            featurizer='ECFP', splitter='random'
                        )
                    elif prop_info['dataset'] == 'lipo':
                        tasks, datasets, transformers = dc.molnet.load_lipo(
                            featurizer='ECFP', splitter='random'
                        )
                    elif prop_info['dataset'] == 'bbbp':
                        tasks, datasets, transformers = dc.molnet.load_bbbp(
                            featurizer='ECFP', splitter='random'
                        )
                    elif prop_info['dataset'] == 'hiv':
                        tasks, datasets, transformers = dc.molnet.load_hiv(
                            featurizer='ECFP', splitter='random'
                        )
                    else:
                        continue
                    train_dataset, valid_dataset, test_dataset = datasets
                    self.models[prop_name] = {
                        'model': model,
                        'transformer': transformers[0] if transformers else None,
                        'task_name': prop_info['task_name'],
                        'type': prop_info['type'],
                        'train_samples': len(train_dataset)
                    }
                    self.property_info[prop_name] = prop_info
                    print(f"✓ {prop_name} model loaded from disk")
                    continue
                # Otherwise, train and save
                print(f"Training {prop_name} model...")
                if prop_info['dataset'] == 'delaney':
                    tasks, datasets, transformers = dc.molnet.load_delaney(
                        featurizer='ECFP', splitter='random'
                    )
                elif prop_info['dataset'] == 'lipo':
                    tasks, datasets, transformers = dc.molnet.load_lipo(
                        featurizer='ECFP', splitter='random'
                    )
                elif prop_info['dataset'] == 'bbbp':
                    tasks, datasets, transformers = dc.molnet.load_bbbp(
                        featurizer='ECFP', splitter='random'
                    )
                elif prop_info['dataset'] == 'hiv':
                    tasks, datasets, transformers = dc.molnet.load_hiv(
                        featurizer='ECFP', splitter='random'
                    )
                else:
                    continue
                train_dataset, valid_dataset, test_dataset = datasets
                if prop_info['type'] == 'regression':
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                else:
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                X_train = train_dataset.X
                y_train = train_dataset.y.flatten()
                model.fit(X_train, y_train)
                # Save model to disk
                joblib.dump(model, model_path)
                self.models[prop_name] = {
                    'model': model,
                    'transformer': transformers[0] if transformers else None,
                    'task_name': prop_info['task_name'],
                    'type': prop_info['type'],
                    'train_samples': len(train_dataset)
                }
                self.property_info[prop_name] = prop_info
                print(f"✓ {prop_name} model trained and saved to disk")
                print(f"  - Training samples: {len(train_dataset)}")
                print(f"  - Type: {prop_info['type']}")
            except Exception as e:
                print(f"⚠️  Failed to load/train {prop_name} model: {e}")
    
    def extract_morgan_features(self, smiles: str) -> np.ndarray:
        """Extract Morgan fingerprints for prediction."""
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            return None
        
        # Generate Morgan fingerprint
        morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        morgan_array = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(morgan_fp, morgan_array)
        
        return morgan_array.reshape(1, -1)
    
    def predict_property(self, smiles: str, property_name: str) -> Optional[float]:
        """Predict a specific property for a compound."""
        if property_name not in self.models:
            return None
        
        try:
            X_pred = self.extract_morgan_features(smiles)
            if X_pred is None:
                return None
            
            model = self.models[property_name]['model']
            model_type = self.models[property_name]['type']
            
            if model_type == 'regression':
                prediction = model.predict(X_pred)[0]
            else:
                # For classification, get probability of positive class
                prediction = model.predict_proba(X_pred)[0, 1]
            
            return prediction
            
        except Exception as e:
            print(f"Prediction failed for {property_name}: {e}")
            return None
    
    def predict_all_properties(self, smiles: str) -> Dict:
        """
        Predict all available properties for a PFAS compound.
        
        Args:
            smiles: SMILES string of the compound
            
        Returns:
            Dictionary containing all property predictions
        """
        print(f"Predicting MoleculeNet properties for: {smiles}")
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {'error': 'Invalid SMILES string'}
        
        properties = {}
        
        # Predict each available property
        for prop_name, prop_info in self.available_properties.items():
            try:
                prediction = self.predict_property(smiles, prop_name)
                
                if prediction is not None:
                    properties[prop_name] = {
                        'value': prediction,
                        'description': prop_info['description'],
                        'unit': prop_info['unit'],
                        'type': prop_info['type'],
                        'method': prop_info['method']
                    }
                    print(f"✓ {prop_name}: {prediction:.4f} {prop_info['unit']}")
                else:
                    print(f"⚠️  {prop_name}: Prediction failed")
                    
            except Exception as e:
                print(f"⚠️  {prop_name}: Error - {e}")
        
        return {
            'smiles': smiles,
            'properties': properties,
            'model_info': {
                'total_models': len(self.models),
                'available_properties': list(self.models.keys())
            }
        }
    
    def evaluate_model_performance(self) -> Dict:
        """Evaluate all models on their respective test datasets."""
        print("Evaluating MoleculeNet model performance...")
        
        evaluation_results = {}
        
        for prop_name, model_info in self.models.items():
            try:
                print(f"Evaluating {prop_name} model...")
                
                # Load test dataset
                prop_info = self.available_properties[prop_name]
                
                if prop_info['dataset'] == 'delaney':
                    tasks, datasets, transformers = dc.molnet.load_delaney(
                        featurizer='ECFP', splitter='random'
                    )
                elif prop_info['dataset'] == 'lipo':
                    tasks, datasets, transformers = dc.molnet.load_lipo(
                        featurizer='ECFP', splitter='random'
                    )
                elif prop_info['dataset'] == 'bbbp':
                    tasks, datasets, transformers = dc.molnet.load_bbbp(
                        featurizer='ECFP', splitter='random'
                    )
                elif prop_info['dataset'] == 'hiv':
                    tasks, datasets, transformers = dc.molnet.load_hiv(
                        featurizer='ECFP', splitter='random'
                    )
                
                train_dataset, valid_dataset, test_dataset = datasets
                
                model = model_info['model']
                X_test = test_dataset.X
                y_test = test_dataset.y.flatten()
                
                if model_info['type'] == 'regression':
                    y_pred = model.predict(X_test)
                    
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, y_pred)
                    
                    metrics = {
                        'mse': mse,
                        'rmse': rmse,
                        'r2': r2,
                        'n_samples': len(y_test)
                    }
                    
                else:
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    y_pred_class = model.predict(X_test)
                    
                    from sklearn.metrics import accuracy_score, roc_auc_score
                    accuracy = accuracy_score(y_test, y_pred_class)
                    auc = roc_auc_score(y_test, y_pred_proba)
                    
                    metrics = {
                        'accuracy': accuracy,
                        'auc': auc,
                        'n_samples': len(y_test)
                    }
                
                evaluation_results[prop_name] = metrics
                print(f"✓ {prop_name} evaluation completed")
                
            except Exception as e:
                print(f"⚠️  {prop_name} evaluation failed: {e}")
        
        return evaluation_results
    
    def print_prediction_report(self, results: Dict):
        """Print a formatted report for MoleculeNet predictions."""
        if 'error' in results:
            print(f"❌ Error: {results['error']}")
            return
        
        print(f"\n{'='*70}")
        print(f"MOLECULENET PFAS PROPERTY PREDICTION REPORT")
        print(f"{'='*70}")
        print(f"SMILES: {results['smiles']}")
        print(f"{'='*70}")
        
        if not results['properties']:
            print("No properties predicted successfully.")
            return
        
        for prop_name, prop_data in results['properties'].items():
            print(f"\n{prop_name.upper()}:")
            print(f"  Description: {prop_data['description']}")
            print(f"  Predicted Value: {prop_data['value']:.4f} {prop_data['unit']}")
            print(f"  Type: {prop_data['type']}")
            print(f"  Method: {prop_data['method']}")
            
            # Add interpretation
            if prop_name == 'solubility':
                if prop_data['value'] > -2:
                    interpretation = "High solubility"
                elif prop_data['value'] > -4:
                    interpretation = "Moderate solubility"
                else:
                    interpretation = "Low solubility"
                print(f"  Interpretation: {interpretation}")
                
            elif prop_name == 'lipophilicity':
                if prop_data['value'] < 2:
                    interpretation = "Low lipophilicity (good for drug-like properties)"
                elif prop_data['value'] < 5:
                    interpretation = "Moderate lipophilicity"
                else:
                    interpretation = "High lipophilicity (may have bioavailability issues)"
                print(f"  Interpretation: {interpretation}")
                
            elif prop_name in ['bbbp', 'hiv']:
                if prop_data['value'] > 0.7:
                    interpretation = "High probability"
                elif prop_data['value'] > 0.3:
                    interpretation = "Moderate probability"
                else:
                    interpretation = "Low probability"
                print(f"  Interpretation: {interpretation}")
        
        print(f"\nModel Information:")
        print(f"  Total models loaded: {results['model_info']['total_models']}")
        print(f"  Available properties: {', '.join(results['model_info']['available_properties'])}")
        
        print(f"\n{'='*70}")


def main():
    """Demo the MoleculeNet PFAS predictor."""
    print("MoleculeNet PFAS Property Predictor Demo")
    print("=" * 60)
    
    # Initialize predictor
    predictor = MoleculeNetPFASPredictor()
    
    # Test with PFAS compounds
    pfas_compounds = {
        "PFOA": "C(=O)(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O",
        "PFOS": "C(C(C(C(C(F)(F)S(=O)(=O)O)(F)F)(F)F)(F)F)(C(C(C(F)(F)F)(F)F)(F)F)(F)F",
    }
    
    print(f"\nAvailable properties: {list(predictor.available_properties.keys())}")
    
    # Predict all properties for each compound
    for name, smiles in pfas_compounds.items():
        print(f"\n{'='*50}")
        print(f"Analyzing {name} with MoleculeNet...")
        print(f"{'='*50}")
        
        # Get predictions
        results = predictor.predict_all_properties(smiles)
        
        # Print detailed report
        predictor.print_prediction_report(results)
    
    # Evaluate model performance
    print(f"\n{'='*50}")
    print("MODEL PERFORMANCE EVALUATION")
    print(f"{'='*50}")
    
    evaluation_results = predictor.evaluate_model_performance()
    
    for prop_name, metrics in evaluation_results.items():
        print(f"\n{prop_name.upper()} Model Performance:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    print(f"\nDemo completed successfully! 🎉")


if __name__ == "__main__":
    main() 
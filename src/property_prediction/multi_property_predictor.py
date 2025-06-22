"""
Multi-Property PFAS Predictor

A comprehensive property prediction system for PFAS compounds using
multiple MoleculeNet datasets and models.

Author: PFAS AI Project
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Core imports
import deepchem as dc
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit import DataStructs
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt


class MultiPropertyPFASPredictor:
    """
    Multi-property predictor for PFAS compounds using MoleculeNet datasets.
    
    Supported properties:
    - Solubility (Delaney/ESOL dataset)
    - Lipophilicity (Lipophilicity dataset)
    - Blood-Brain Barrier Penetration (BBBP dataset)
    - HIV replication inhibition (HIV dataset)
    """
    
    def __init__(self):
        """Initialize the multi-property predictor."""
        self.models = {}
        self.datasets = {}
        self.property_info = {}
        
        # Define available properties and their datasets
        self.available_properties = {
            'solubility': {
                'dataset': 'delaney',
                'task_name': 'measured log solubility in mols per litre',
                'type': 'regression',
                'description': 'Aqueous solubility (logS)',
                'unit': 'log(mol/L)'
            },
            'lipophilicity': {
                'dataset': 'lipo',
                'task_name': 'exp',
                'type': 'regression',
                'description': 'Octanol-water partition coefficient',
                'unit': 'logP'
            },
            'bbbp': {
                'dataset': 'bbbp',
                'task_name': 'p_np',
                'type': 'classification',
                'description': 'Blood-brain barrier penetration',
                'unit': 'probability'
            },
            'hiv': {
                'dataset': 'hiv',
                'task_name': 'HIV_active',
                'type': 'classification',
                'description': 'HIV replication inhibition',
                'unit': 'probability'
            }
        }
        
        # Load models
        self._load_all_models()
    
    def _load_all_models(self):
        """Load all available MoleculeNet models."""
        print("Loading MoleculeNet models for multiple properties...")
        
        for prop_name, prop_info in self.available_properties.items():
            try:
                print(f"Loading {prop_name} model...")
                
                # Load dataset
                if prop_info['dataset'] == 'delaney':
                    tasks, datasets, transformers = dc.molnet.load_delaney(
                        featurizer='ECFP', split='random'
                    )
                elif prop_info['dataset'] == 'lipo':
                    tasks, datasets, transformers = dc.molnet.load_lipo(
                        featurizer='ECFP', split='random'
                    )
                elif prop_info['dataset'] == 'bbbp':
                    tasks, datasets, transformers = dc.molnet.load_bbbp(
                        featurizer='ECFP', split='random'
                    )
                elif prop_info['dataset'] == 'hiv':
                    tasks, datasets, transformers = dc.molnet.load_hiv(
                        featurizer='ECFP', split='random'
                    )
                else:
                    continue
                
                train_dataset, valid_dataset, test_dataset = datasets
                
                # Train model
                if prop_info['type'] == 'regression':
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                else:
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                
                X_train = train_dataset.X
                y_train = train_dataset.y.flatten()
                
                model.fit(X_train, y_train)
                
                # Store model and info
                self.models[prop_name] = {
                    'model': model,
                    'transformer': transformers[0] if transformers else None,
                    'task_name': prop_info['task_name'],
                    'type': prop_info['type']
                }
                
                self.datasets[prop_name] = {
                    'train': train_dataset,
                    'valid': valid_dataset,
                    'test': test_dataset
                }
                
                self.property_info[prop_name] = prop_info
                
                print(f"✓ {prop_name} model loaded successfully")
                print(f"  - Training samples: {len(train_dataset)}")
                print(f"  - Type: {prop_info['type']}")
                
            except Exception as e:
                print(f"⚠️  Failed to load {prop_name} model: {e}")
    
    def extract_molecular_features(self, smiles_list: List[str]) -> Dict:
        """
        Extract comprehensive molecular features using RDKit.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Dictionary containing different feature types
        """
        features = {
            'basic': [],
            'rdkit': [],
            'morgan': [],
            'pfas_specific': []
        }
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is None:
                # Add zeros for invalid molecules
                features['basic'].append([0] * 10)
                features['rdkit'].append([0] * 200)
                features['morgan'].append([0] * 2048)
                features['pfas_specific'].append([0] * 8)
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
            
            # PFAS-specific features
            pfas_feats = self._extract_pfas_features(mol)
            features['pfas_specific'].append(pfas_feats)
        
        # Convert to numpy arrays
        for key in features:
            features[key] = np.array(features[key])
        
        return features
    
    def _extract_pfas_features(self, mol) -> List[float]:
        """Extract PFAS-specific molecular features."""
        features = []
        
        # Count fluorine atoms
        fluorine_count = 0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'F':
                fluorine_count += 1
        features.append(fluorine_count)
        
        # Fluorine ratio
        total_atoms = mol.GetNumAtoms()
        fluorine_ratio = fluorine_count / total_atoms if total_atoms > 0 else 0
        features.append(fluorine_ratio)
        
        # Count perfluoroalkyl groups (simplified)
        perfluoro_pattern = Chem.MolFromSmarts('[C]([F])([F])[F]')
        perfluoro_matches = mol.GetSubstructMatches(perfluoro_pattern)
        features.append(len(perfluoro_matches))
        
        # Count sulfonic acid groups
        sulfonic_pattern = Chem.MolFromSmarts('[S](=O)(=O)[O-]')
        sulfonic_matches = mol.GetSubstructMatches(sulfonic_pattern)
        features.append(len(sulfonic_matches))
        
        # Count carboxylic acid groups
        carboxylic_pattern = Chem.MolFromSmarts('[C](=O)[O-]')
        carboxylic_matches = mol.GetSubstructMatches(carboxylic_pattern)
        features.append(len(carboxylic_matches))
        
        # Chain length (approximate)
        chain_length = Chem.Descriptors.HeavyAtomCount(mol)
        features.append(chain_length)
        
        # Branching factor
        branching = Chem.Descriptors.NumRotatableBonds(mol)
        features.append(branching)
        
        # Polarity index
        polarity = Chem.Descriptors.TPSA(mol) / Chem.Descriptors.MolWt(mol) if Chem.Descriptors.MolWt(mol) > 0 else 0
        features.append(polarity)
        
        return features
    
    def predict_all_properties(self, smiles_list: List[str]) -> Dict:
        """
        Predict all available properties for PFAS compounds.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Dictionary containing predictions for all properties
        """
        print(f"Predicting all properties for {len(smiles_list)} compounds...")
        
        # Extract features
        features = self.extract_molecular_features(smiles_list)
        
        all_predictions = {}
        
        # Make predictions for each property
        for prop_name, model_info in self.models.items():
            try:
                print(f"Predicting {prop_name}...")
                
                model = model_info['model']
                X_pred = features['morgan']  # Use Morgan fingerprints for prediction
                
                if model_info['type'] == 'regression':
                    predictions = model.predict(X_pred)
                else:
                    # For classification, get probability of positive class
                    predictions = model.predict_proba(X_pred)[:, 1]
                
                all_predictions[prop_name] = {
                    'predictions': predictions,
                    'type': model_info['type'],
                    'info': self.property_info[prop_name]
                }
                
                print(f"✓ {prop_name} predictions completed")
                
            except Exception as e:
                print(f"⚠️  {prop_name} prediction failed: {e}")
        
        return {
            'smiles': smiles_list,
            'predictions': all_predictions,
            'features': features
        }
    
    def predict_single_compound(self, smiles: str) -> Dict:
        """
        Predict all properties for a single PFAS compound.
        
        Args:
            smiles: SMILES string of the compound
            
        Returns:
            Dictionary containing all property predictions
        """
        results = self.predict_all_properties([smiles])
        
        # Extract single compound results
        single_results = {
            'smiles': smiles,
            'properties': {}
        }
        
        for prop_name, prop_data in results['predictions'].items():
            single_results['properties'][prop_name] = {
                'value': prop_data['predictions'][0],
                'type': prop_data['type'],
                'description': prop_data['info']['description'],
                'unit': prop_data['info']['unit']
            }
        
        return single_results
    
    def evaluate_all_models(self) -> Dict:
        """Evaluate all models on their respective test datasets."""
        print("Evaluating all models...")
        
        evaluation_results = {}
        
        for prop_name, model_info in self.models.items():
            try:
                print(f"Evaluating {prop_name} model...")
                
                model = model_info['model']
                test_dataset = self.datasets[prop_name]['test']
                
                X_test = test_dataset.X
                y_test = test_dataset.y.flatten()
                
                if model_info['type'] == 'regression':
                    y_pred = model.predict(X_test)
                    
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    metrics = {
                        'mse': mse,
                        'rmse': rmse,
                        'mae': mae,
                        'r2': r2,
                        'n_samples': len(y_test)
                    }
                    
                else:
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    y_pred_class = model.predict(X_test)
                    
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
    
    def print_compound_report(self, compound_results: Dict):
        """Print a formatted report for a single compound."""
        print(f"\n{'='*60}")
        print(f"PFAS COMPOUND ANALYSIS REPORT")
        print(f"{'='*60}")
        print(f"SMILES: {compound_results['smiles']}")
        print(f"{'='*60}")
        
        for prop_name, prop_data in compound_results['properties'].items():
            print(f"\n{prop_name.upper()}:")
            print(f"  Description: {prop_data['description']}")
            print(f"  Predicted Value: {prop_data['value']:.4f} {prop_data['unit']}")
            print(f"  Type: {prop_data['type']}")
            
            # Add interpretation for key properties
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
        
        print(f"\n{'='*60}")


def main():
    """Demo the multi-property PFAS predictor."""
    print("Multi-Property PFAS Predictor Demo")
    print("=" * 50)
    
    # Initialize predictor
    predictor = MultiPropertyPFASPredictor()
    
    # Test with PFAS compounds
    pfas_compounds = [
        "C(=O)(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O",  # PFOA
        "C(C(C(C(C(F)(F)S(=O)(=O)O)(F)F)(F)F)(F)F)(C(C(C(F)(F)F)(F)F)(F)F)(F)F",  # PFOS
        "C(=O)(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)O",  # PFHxA
        "C(C(C(F)(F)S(=O)(=O)O)(F)F)(C(F)(F)F)(F)F",  # PFBS
    ]
    
    pfas_names = ["PFOA", "PFOS", "PFHxA", "PFBS"]
    
    print(f"\nAvailable properties: {list(predictor.available_properties.keys())}")
    
    # Predict all properties for each compound
    for i, (smiles, name) in enumerate(zip(pfas_compounds, pfas_names)):
        print(f"\n{'='*40}")
        print(f"Analyzing {name}...")
        print(f"{'='*40}")
        
        # Get predictions for single compound
        results = predictor.predict_single_compound(smiles)
        
        # Print detailed report
        predictor.print_compound_report(results)
    
    # Evaluate all models
    print(f"\n{'='*50}")
    print("MODEL EVALUATION")
    print(f"{'='*50}")
    
    evaluation_results = predictor.evaluate_all_models()
    
    for prop_name, metrics in evaluation_results.items():
        print(f"\n{prop_name.upper()} Model Performance:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    print(f"\nDemo completed successfully! 🎉")


if __name__ == "__main__":
    main() 
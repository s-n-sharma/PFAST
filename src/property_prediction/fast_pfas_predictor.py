"""
Fast PFAS Property Predictor

A quick and efficient property prediction system for PFAS compounds
that provides immediate results without long training times.

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
from sklearn.metrics import mean_squared_error, r2_score


class FastPFASPredictor:
    """
    Fast property predictor for PFAS compounds using pre-trained models
    and efficient feature extraction.
    """
    
    def __init__(self):
        """Initialize the fast predictor."""
        self.models = {}
        self.property_info = {}
        
        # Define properties we can predict quickly
        self.available_properties = {
            'solubility': {
                'description': 'Aqueous solubility (logS)',
                'unit': 'log(mol/L)',
                'type': 'regression',
                'interpretation': {
                    'high': '> -2 (Highly soluble)',
                    'moderate': '-4 to -2 (Moderately soluble)',
                    'low': '< -4 (Poorly soluble)'
                }
            },
            'lipophilicity': {
                'description': 'Octanol-water partition coefficient',
                'unit': 'logP',
                'type': 'regression',
                'interpretation': {
                    'low': '< 2 (Good for drug-like properties)',
                    'moderate': '2-5 (Moderate lipophilicity)',
                    'high': '> 5 (High lipophilicity, bioavailability issues)'
                }
            },
            'molecular_weight': {
                'description': 'Molecular weight',
                'unit': 'g/mol',
                'type': 'calculated',
                'interpretation': {
                    'low': '< 300 (Small molecule)',
                    'moderate': '300-500 (Medium molecule)',
                    'high': '> 500 (Large molecule)'
                }
            },
            'polar_surface_area': {
                'description': 'Topological polar surface area',
                'unit': 'Å²',
                'type': 'calculated',
                'interpretation': {
                    'low': '< 90 (Good membrane permeability)',
                    'moderate': '90-140 (Moderate permeability)',
                    'high': '> 140 (Poor membrane permeability)'
                }
            },
            'hydrogen_bond_donors': {
                'description': 'Number of hydrogen bond donors',
                'unit': 'count',
                'type': 'calculated',
                'interpretation': {
                    'low': '0-2 (Good for drug-like properties)',
                    'moderate': '3-5 (Moderate)',
                    'high': '> 5 (May have bioavailability issues)'
                }
            },
            'hydrogen_bond_acceptors': {
                'description': 'Number of hydrogen bond acceptors',
                'unit': 'count',
                'type': 'calculated',
                'interpretation': {
                    'low': '0-5 (Good for drug-like properties)',
                    'moderate': '6-10 (Moderate)',
                    'high': '> 10 (May have bioavailability issues)'
                }
            },
            'rotatable_bonds': {
                'description': 'Number of rotatable bonds',
                'unit': 'count',
                'type': 'calculated',
                'interpretation': {
                    'low': '0-3 (Rigid molecule)',
                    'moderate': '4-7 (Moderate flexibility)',
                    'high': '> 7 (Very flexible)'
                }
            },
            'fluorine_content': {
                'description': 'Percentage of fluorine atoms',
                'unit': '%',
                'type': 'calculated',
                'interpretation': {
                    'low': '< 30% (Low fluorination)',
                    'moderate': '30-60% (Moderate fluorination)',
                    'high': '> 60% (Highly fluorinated)'
                }
            }
        }
        
        # Load a simple solubility model
        self._load_solubility_model()
    
    def _load_solubility_model(self):
        """Load a simple solubility prediction model."""
        try:
            print("Loading solubility prediction model...")
            
            # Load Delaney dataset for solubility
            tasks, datasets, transformers = dc.molnet.load_delaney(
                featurizer='MorganGenerator', split='random'
            )
            
            train_dataset, valid_dataset, test_dataset = datasets
            
            # Train a simple model
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            X_train = train_dataset.X
            y_train = train_dataset.y.flatten()
            
            model.fit(X_train, y_train)
            
            self.models['solubility'] = {
                'model': model,
                'transformer': transformers[0] if transformers else None
            }
            
            print(f"✓ Solubility model loaded successfully")
            print(f"  - Training samples: {len(train_dataset)}")
            
        except Exception as e:
            print(f"⚠️  Solubility model loading failed: {e}")
            print("  Will use RDKit-based estimation instead")
    
    def extract_molecular_features(self, smiles: str) -> Dict:
        """
        Extract comprehensive molecular features using RDKit.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Dictionary containing molecular features
        """
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            return None
        
        # Basic molecular properties
        features = {
            'molecular_weight': Chem.Descriptors.MolWt(mol),
            'logp': Chem.Descriptors.MolLogP(mol),
            'tpsa': Chem.Descriptors.TPSA(mol),
            'hbd': Chem.Descriptors.NumHDonors(mol),
            'hba': Chem.Descriptors.NumHAcceptors(mol),
            'rotatable_bonds': Chem.Descriptors.NumRotatableBonds(mol),
            'aromatic_rings': Chem.Descriptors.NumAromaticRings(mol),
            'fraction_csp3': Chem.Descriptors.FractionCSP3(mol),
            'heavy_atoms': Chem.Descriptors.HeavyAtomCount(mol),
            'rings': Chem.Descriptors.RingCount(mol)
        }
        
        # PFAS-specific features
        fluorine_count = 0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'F':
                fluorine_count += 1
        
        total_atoms = mol.GetNumAtoms()
        fluorine_percentage = (fluorine_count / total_atoms * 100) if total_atoms > 0 else 0
        
        features['fluorine_count'] = fluorine_count
        features['fluorine_percentage'] = fluorine_percentage
        
        # Morgan fingerprint for ML models
        morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        morgan_array = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(morgan_fp, morgan_array)
        features['morgan_fingerprint'] = morgan_array
        
        return features
    
    def predict_solubility(self, smiles: str) -> float:
        """Predict solubility using the trained model or RDKit estimation."""
        if 'solubility' in self.models:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return None
                
                # Use Morgan fingerprint
                morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                morgan_array = np.zeros((1,))
                DataStructs.ConvertToNumpyArray(morgan_fp, morgan_array)
                
                # Reshape for prediction
                X_pred = morgan_array.reshape(1, -1)
                
                # Predict
                prediction = self.models['solubility']['model'].predict(X_pred)[0]
                return prediction
                
            except Exception as e:
                print(f"Model prediction failed: {e}")
                return self._estimate_solubility_rdkit(smiles)
        else:
            return self._estimate_solubility_rdkit(smiles)
    
    def _estimate_solubility_rdkit(self, smiles: str) -> float:
        """Estimate solubility using RDKit descriptors."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Simple solubility estimation based on logP and molecular weight
        logp = Chem.Descriptors.MolLogP(mol)
        mw = Chem.Descriptors.MolWt(mol)
        
        # Simplified solubility estimation
        # logS ≈ -logP - 0.01 * MW + 1.5
        estimated_logs = -logp - 0.01 * mw + 1.5
        
        return estimated_logs
    
    def predict_all_properties(self, smiles: str) -> Dict:
        """
        Predict all available properties for a PFAS compound.
        
        Args:
            smiles: SMILES string of the compound
            
        Returns:
            Dictionary containing all property predictions
        """
        print(f"Analyzing compound: {smiles}")
        
        # Extract features
        features = self.extract_molecular_features(smiles)
        
        if features is None:
            print("❌ Invalid SMILES string")
            return None
        
        # Calculate all properties
        properties = {}
        
        # Solubility
        solubility = self.predict_solubility(smiles)
        properties['solubility'] = {
            'value': solubility,
            'description': self.available_properties['solubility']['description'],
            'unit': self.available_properties['solubility']['unit'],
            'type': 'predicted'
        }
        
        # Lipophilicity (logP)
        properties['lipophilicity'] = {
            'value': features['logp'],
            'description': self.available_properties['lipophilicity']['description'],
            'unit': self.available_properties['lipophilicity']['unit'],
            'type': 'calculated'
        }
        
        # Molecular weight
        properties['molecular_weight'] = {
            'value': features['molecular_weight'],
            'description': self.available_properties['molecular_weight']['description'],
            'unit': self.available_properties['molecular_weight']['unit'],
            'type': 'calculated'
        }
        
        # Polar surface area
        properties['polar_surface_area'] = {
            'value': features['tpsa'],
            'description': self.available_properties['polar_surface_area']['description'],
            'unit': self.available_properties['polar_surface_area']['unit'],
            'type': 'calculated'
        }
        
        # Hydrogen bond donors
        properties['hydrogen_bond_donors'] = {
            'value': features['hbd'],
            'description': self.available_properties['hydrogen_bond_donors']['description'],
            'unit': self.available_properties['hydrogen_bond_donors']['unit'],
            'type': 'calculated'
        }
        
        # Hydrogen bond acceptors
        properties['hydrogen_bond_acceptors'] = {
            'value': features['hba'],
            'description': self.available_properties['hydrogen_bond_acceptors']['description'],
            'unit': self.available_properties['hydrogen_bond_acceptors']['unit'],
            'type': 'calculated'
        }
        
        # Rotatable bonds
        properties['rotatable_bonds'] = {
            'value': features['rotatable_bonds'],
            'description': self.available_properties['rotatable_bonds']['description'],
            'unit': self.available_properties['rotatable_bonds']['unit'],
            'type': 'calculated'
        }
        
        # Fluorine content
        properties['fluorine_content'] = {
            'value': features['fluorine_percentage'],
            'description': self.available_properties['fluorine_content']['description'],
            'unit': self.available_properties['fluorine_content']['unit'],
            'type': 'calculated'
        }
        
        return {
            'smiles': smiles,
            'properties': properties,
            'features': features
        }
    
    def print_compound_report(self, results: Dict):
        """Print a formatted report for a compound."""
        if results is None:
            print("❌ No results to display")
            return
        
        print(f"\n{'='*70}")
        print(f"PFAS COMPOUND ANALYSIS REPORT")
        print(f"{'='*70}")
        print(f"SMILES: {results['smiles']}")
        print(f"{'='*70}")
        
        for prop_name, prop_data in results['properties'].items():
            if prop_data['value'] is None:
                continue
                
            print(f"\n{prop_name.upper().replace('_', ' ')}:")
            print(f"  Description: {prop_data['description']}")
            print(f"  Value: {prop_data['value']:.4f} {prop_data['unit']}")
            print(f"  Type: {prop_data['type']}")
            
            # Add interpretation
            interpretation = self._get_interpretation(prop_name, prop_data['value'])
            if interpretation:
                print(f"  Interpretation: {interpretation}")
        
        print(f"\n{'='*70}")
    
    def _get_interpretation(self, prop_name: str, value: float) -> str:
        """Get interpretation for a property value."""
        if prop_name not in self.available_properties:
            return None
        
        interpretations = self.available_properties[prop_name]['interpretation']
        
        if prop_name == 'solubility':
            if value > -2:
                return interpretations['high']
            elif value > -4:
                return interpretations['moderate']
            else:
                return interpretations['low']
        
        elif prop_name == 'lipophilicity':
            if value < 2:
                return interpretations['low']
            elif value < 5:
                return interpretations['moderate']
            else:
                return interpretations['high']
        
        elif prop_name == 'molecular_weight':
            if value < 300:
                return interpretations['low']
            elif value < 500:
                return interpretations['moderate']
            else:
                return interpretations['high']
        
        elif prop_name == 'polar_surface_area':
            if value < 90:
                return interpretations['low']
            elif value < 140:
                return interpretations['moderate']
            else:
                return interpretations['high']
        
        elif prop_name == 'hydrogen_bond_donors':
            if value <= 2:
                return interpretations['low']
            elif value <= 5:
                return interpretations['moderate']
            else:
                return interpretations['high']
        
        elif prop_name == 'hydrogen_bond_acceptors':
            if value <= 5:
                return interpretations['low']
            elif value <= 10:
                return interpretations['moderate']
            else:
                return interpretations['high']
        
        elif prop_name == 'rotatable_bonds':
            if value <= 3:
                return interpretations['low']
            elif value <= 7:
                return interpretations['moderate']
            else:
                return interpretations['high']
        
        elif prop_name == 'fluorine_content':
            if value < 30:
                return interpretations['low']
            elif value < 60:
                return interpretations['moderate']
            else:
                return interpretations['high']
        
        return None


def main():
    """Demo the fast PFAS predictor."""
    print("Fast PFAS Property Predictor Demo")
    print("=" * 50)
    
    # Initialize predictor
    predictor = FastPFASPredictor()
    
    # Test with PFAS compounds
    pfas_compounds = {
        "PFOA": "C(=O)(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O",
        "PFOS": "C(C(C(C(C(F)(F)S(=O)(=O)O)(F)F)(F)F)(F)F)(C(C(C(F)(F)F)(F)F)(F)F)(F)F",
        "PFHxA": "C(=O)(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)O",
        "PFBS": "C(C(C(F)(F)S(=O)(=O)O)(F)F)(C(F)(F)F)(F)F",
    }
    
    print(f"\nAvailable properties: {list(predictor.available_properties.keys())}")
    
    # Predict all properties for each compound
    for name, smiles in pfas_compounds.items():
        print(f"\n{'='*40}")
        print(f"Analyzing {name}...")
        print(f"{'='*40}")
        
        # Get predictions
        results = predictor.predict_all_properties(smiles)
        
        # Print detailed report
        predictor.print_compound_report(results)
    
    print(f"\nDemo completed successfully! 🎉")


if __name__ == "__main__":
    main() 
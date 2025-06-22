"""
Enhanced Fast PFAS Property Predictor

A comprehensive and fast property prediction system for PFAS compounds
that provides immediate results with multiple property predictions.

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
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score


class EnhancedPFASPredictor:
    """
    Enhanced property predictor for PFAS compounds with multiple property predictions.
    
    Supported properties:
    - Solubility (aqueous)
    - Lipophilicity (logP)
    - Molecular weight
    - Polar surface area
    - Hydrogen bonding
    - Rotatable bonds
    - Fluorine content
    - Drug-likeness
    - Bioaccumulation potential
    - Environmental persistence
    - Toxicity indicators
    - Membrane permeability
    - Metabolic stability
    """
    
    def __init__(self):
        """Initialize the enhanced predictor."""
        self.models = {}
        self.property_info = {}
        
        # Define comprehensive property set
        self.available_properties = {
            'solubility': {
                'description': 'Aqueous solubility (logS)',
                'unit': 'log(mol/L)',
                'type': 'predicted',
                'interpretation': {
                    'high': '> -2 (Highly soluble)',
                    'moderate': '-4 to -2 (Moderately soluble)',
                    'low': '< -4 (Poorly soluble)'
                }
            },
            'lipophilicity': {
                'description': 'Octanol-water partition coefficient',
                'unit': 'logP',
                'type': 'calculated',
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
            },
            'fluorine_count': {
                'description': 'Number of fluorine atoms',
                'unit': 'count',
                'type': 'calculated',
                'interpretation': {
                    'low': '0-5 (Low fluorination)',
                    'moderate': '6-15 (Moderate fluorination)',
                    'high': '> 15 (Highly fluorinated)'
                }
            },
            'drug_likeness': {
                'description': 'Lipinski drug-likeness score',
                'unit': 'score',
                'type': 'calculated',
                'interpretation': {
                    'good': '≥ 4 (Good drug-like properties)',
                    'moderate': '2-3 (Moderate drug-like properties)',
                    'poor': '0-1 (Poor drug-like properties)'
                }
            },
            'bioaccumulation_potential': {
                'description': 'Bioaccumulation potential score',
                'unit': 'score',
                'type': 'calculated',
                'interpretation': {
                    'low': '0-2 (Low bioaccumulation)',
                    'moderate': '3-4 (Moderate bioaccumulation)',
                    'high': '5-6 (High bioaccumulation)'
                }
            },
            'environmental_persistence': {
                'description': 'Environmental persistence score',
                'unit': 'score',
                'type': 'calculated',
                'interpretation': {
                    'low': '0-2 (Low persistence)',
                    'moderate': '3-4 (Moderate persistence)',
                    'high': '5-6 (High persistence)'
                }
            },
            'toxicity_risk': {
                'description': 'Toxicity risk assessment',
                'unit': 'score',
                'type': 'calculated',
                'interpretation': {
                    'low': '0-2 (Low toxicity risk)',
                    'moderate': '3-4 (Moderate toxicity risk)',
                    'high': '5-6 (High toxicity risk)'
                }
            },
            'membrane_permeability': {
                'description': 'Membrane permeability prediction',
                'unit': 'score',
                'type': 'calculated',
                'interpretation': {
                    'low': '0-2 (Poor permeability)',
                    'moderate': '3-4 (Moderate permeability)',
                    'high': '5-6 (Good permeability)'
                }
            },
            'metabolic_stability': {
                'description': 'Metabolic stability prediction',
                'unit': 'score',
                'type': 'calculated',
                'interpretation': {
                    'low': '0-2 (Unstable)',
                    'moderate': '3-4 (Moderate stability)',
                    'high': '5-6 (Stable)'
                }
            },
            'water_solubility_class': {
                'description': 'Water solubility classification',
                'unit': 'class',
                'type': 'calculated',
                'interpretation': {
                    'highly_soluble': '> 10 g/L',
                    'soluble': '1-10 g/L',
                    'sparingly_soluble': '0.1-1 g/L',
                    'slightly_soluble': '0.01-0.1 g/L',
                    'very_slightly_soluble': '< 0.01 g/L'
                }
            },
            'vapor_pressure': {
                'description': 'Estimated vapor pressure',
                'unit': 'Pa',
                'type': 'calculated',
                'interpretation': {
                    'high': '> 1000 (Volatile)',
                    'moderate': '10-1000 (Moderate volatility)',
                    'low': '< 10 (Low volatility)'
                }
            }
        }
        
        # Load solubility model
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
            'rings': Chem.Descriptors.RingCount(mol),
            'saturated_rings': Chem.Descriptors.NumSaturatedRings(mol),
            'aliphatic_rings': Chem.Descriptors.NumAliphaticRings(mol),
            'heteroatoms': Chem.Descriptors.NumHeteroatoms(mol),
            'spiro_atoms': Chem.Descriptors.NumSpiroAtoms(mol),
            'bridgehead_atoms': Chem.Descriptors.NumBridgeheadAtoms(mol),
        }
        
        # Use rdMolDescriptors for bond counting (these don't exist in Chem.Descriptors)
        try:
            features['amide_bonds'] = rdMolDescriptors.CalcNumAmideBonds(mol)
        except:
            features['amide_bonds'] = 0
            
        try:
            features['ester_bonds'] = rdMolDescriptors.CalcNumEsterBonds(mol)
        except:
            features['ester_bonds'] = 0
            
        try:
            features['ether_bonds'] = rdMolDescriptors.CalcNumEtherBonds(mol)
        except:
            features['ether_bonds'] = 0
            
        try:
            features['sulfonamides'] = rdMolDescriptors.CalcNumSulfonamides(mol)
        except:
            features['sulfonamides'] = 0
            
        try:
            features['sulfones'] = rdMolDescriptors.CalcNumSulfones(mol)
        except:
            features['sulfones'] = 0
        
        # Ring system descriptors using rdMolDescriptors
        try:
            features['aliphatic_carbocycles'] = rdMolDescriptors.CalcNumAliphaticCarbocycles(mol)
        except:
            features['aliphatic_carbocycles'] = 0
            
        try:
            features['aliphatic_heterocycles'] = rdMolDescriptors.CalcNumAliphaticHeterocycles(mol)
        except:
            features['aliphatic_heterocycles'] = 0
            
        try:
            features['aromatic_carbocycles'] = rdMolDescriptors.CalcNumAromaticCarbocycles(mol)
        except:
            features['aromatic_carbocycles'] = 0
            
        try:
            features['aromatic_heterocycles'] = rdMolDescriptors.CalcNumAromaticHeterocycles(mol)
        except:
            features['aromatic_heterocycles'] = 0
            
        try:
            features['saturated_carbocycles'] = rdMolDescriptors.CalcNumSaturatedCarbocycles(mol)
        except:
            features['saturated_carbocycles'] = 0
            
        try:
            features['saturated_heterocycles'] = rdMolDescriptors.CalcNumSaturatedHeterocycles(mol)
        except:
            features['saturated_heterocycles'] = 0
        
        # PFAS-specific features
        fluorine_count = 0
        carbon_count = 0
        oxygen_count = 0
        sulfur_count = 0
        
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol == 'F':
                fluorine_count += 1
            elif symbol == 'C':
                carbon_count += 1
            elif symbol == 'O':
                oxygen_count += 1
            elif symbol == 'S':
                sulfur_count += 1
        
        total_atoms = mol.GetNumAtoms()
        fluorine_percentage = (fluorine_count / total_atoms * 100) if total_atoms > 0 else 0
        
        features['fluorine_count'] = fluorine_count
        features['fluorine_percentage'] = fluorine_percentage
        features['carbon_count'] = carbon_count
        features['oxygen_count'] = oxygen_count
        features['sulfur_count'] = sulfur_count
        
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
        
        # Enhanced solubility estimation
        logp = Chem.Descriptors.MolLogP(mol)
        mw = Chem.Descriptors.MolWt(mol)
        tpsa = Chem.Descriptors.TPSA(mol)
        
        # More sophisticated estimation
        estimated_logs = -logp - 0.01 * mw + 0.01 * tpsa + 1.5
        
        return estimated_logs
    
    def calculate_advanced_properties(self, features: Dict) -> Dict:
        """Calculate advanced properties based on molecular features."""
        advanced_props = {}
        
        # Drug-likeness score (Lipinski rules)
        drug_score = 0
        if features['molecular_weight'] <= 500:
            drug_score += 1
        if features['logp'] <= 5:
            drug_score += 1
        if features['hbd'] <= 5:
            drug_score += 1
        if features['hba'] <= 10:
            drug_score += 1
        
        advanced_props['drug_likeness'] = drug_score
        
        # Bioaccumulation potential
        bioaccum_score = 0
        if features['logp'] > 3:
            bioaccum_score += 2
        elif features['logp'] > 1:
            bioaccum_score += 1
        
        if features['molecular_weight'] > 400:
            bioaccum_score += 1
        
        if features['fluorine_percentage'] > 50:
            bioaccum_score += 2
        elif features['fluorine_percentage'] > 20:
            bioaccum_score += 1
        
        advanced_props['bioaccumulation_potential'] = min(bioaccum_score, 6)
        
        # Environmental persistence
        persistence_score = 0
        if features['fluorine_percentage'] > 60:
            persistence_score += 3
        elif features['fluorine_percentage'] > 30:
            persistence_score += 2
        
        if features['logp'] > 4:
            persistence_score += 1
        
        if features['molecular_weight'] > 300:
            persistence_score += 1
        
        if features['sulfur_count'] > 0:
            persistence_score += 1
        
        advanced_props['environmental_persistence'] = min(persistence_score, 6)
        
        # Toxicity risk
        toxicity_score = 0
        if features['fluorine_percentage'] > 50:
            toxicity_score += 2
        
        if features['logp'] > 5:
            toxicity_score += 1
        
        if features['molecular_weight'] > 500:
            toxicity_score += 1
        
        if features['sulfur_count'] > 0:
            toxicity_score += 1
        
        if features['oxygen_count'] > 4:
            toxicity_score += 1
        
        advanced_props['toxicity_risk'] = min(toxicity_score, 6)
        
        # Membrane permeability
        permeability_score = 0
        if features['tpsa'] < 90:
            permeability_score += 2
        elif features['tpsa'] < 140:
            permeability_score += 1
        
        if features['logp'] > 1 and features['logp'] < 5:
            permeability_score += 2
        elif features['logp'] > 0:
            permeability_score += 1
        
        if features['molecular_weight'] < 500:
            permeability_score += 1
        
        if features['rotatable_bonds'] < 5:
            permeability_score += 1
        
        advanced_props['membrane_permeability'] = min(permeability_score, 6)
        
        # Metabolic stability
        stability_score = 0
        if features['fluorine_percentage'] > 40:
            stability_score += 2
        
        if features['logp'] > 2:
            stability_score += 1
        
        if features['molecular_weight'] > 300:
            stability_score += 1
        
        if features['aromatic_rings'] > 0:
            stability_score += 1
        
        if features['sulfur_count'] > 0:
            stability_score += 1
        
        advanced_props['metabolic_stability'] = min(stability_score, 6)
        
        # Water solubility class
        estimated_sol = -features['logp'] - 0.01 * features['molecular_weight'] + 1.5
        if estimated_sol > 0:
            solubility_class = "highly_soluble"
        elif estimated_sol > -2:
            solubility_class = "soluble"
        elif estimated_sol > -3:
            solubility_class = "sparingly_soluble"
        elif estimated_sol > -4:
            solubility_class = "slightly_soluble"
        else:
            solubility_class = "very_slightly_soluble"
        
        advanced_props['water_solubility_class'] = solubility_class
        
        # Vapor pressure estimation
        # Simplified estimation based on molecular weight and logP
        vapor_pressure = 10**(3 - features['logp'] - 0.01 * features['molecular_weight'])
        advanced_props['vapor_pressure'] = vapor_pressure
        
        return advanced_props
    
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
        
        # Basic properties
        properties['solubility'] = {
            'value': self.predict_solubility(smiles),
            'description': self.available_properties['solubility']['description'],
            'unit': self.available_properties['solubility']['unit'],
            'type': 'predicted'
        }
        
        properties['lipophilicity'] = {
            'value': features['logp'],
            'description': self.available_properties['lipophilicity']['description'],
            'unit': self.available_properties['lipophilicity']['unit'],
            'type': 'calculated'
        }
        
        properties['molecular_weight'] = {
            'value': features['molecular_weight'],
            'description': self.available_properties['molecular_weight']['description'],
            'unit': self.available_properties['molecular_weight']['unit'],
            'type': 'calculated'
        }
        
        properties['polar_surface_area'] = {
            'value': features['tpsa'],
            'description': self.available_properties['polar_surface_area']['description'],
            'unit': self.available_properties['polar_surface_area']['unit'],
            'type': 'calculated'
        }
        
        properties['hydrogen_bond_donors'] = {
            'value': features['hbd'],
            'description': self.available_properties['hydrogen_bond_donors']['description'],
            'unit': self.available_properties['hydrogen_bond_donors']['unit'],
            'type': 'calculated'
        }
        
        properties['hydrogen_bond_acceptors'] = {
            'value': features['hba'],
            'description': self.available_properties['hydrogen_bond_acceptors']['description'],
            'unit': self.available_properties['hydrogen_bond_acceptors']['unit'],
            'type': 'calculated'
        }
        
        properties['rotatable_bonds'] = {
            'value': features['rotatable_bonds'],
            'description': self.available_properties['rotatable_bonds']['description'],
            'unit': self.available_properties['rotatable_bonds']['unit'],
            'type': 'calculated'
        }
        
        properties['fluorine_content'] = {
            'value': features['fluorine_percentage'],
            'description': self.available_properties['fluorine_content']['description'],
            'unit': self.available_properties['fluorine_content']['unit'],
            'type': 'calculated'
        }
        
        properties['fluorine_count'] = {
            'value': features['fluorine_count'],
            'description': self.available_properties['fluorine_count']['description'],
            'unit': self.available_properties['fluorine_count']['unit'],
            'type': 'calculated'
        }
        
        # Advanced properties
        advanced_props = self.calculate_advanced_properties(features)
        
        for prop_name, value in advanced_props.items():
            properties[prop_name] = {
                'value': value,
                'description': self.available_properties[prop_name]['description'],
                'unit': self.available_properties[prop_name]['unit'],
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
        
        print(f"\n{'='*80}")
        print(f"ENHANCED PFAS COMPOUND ANALYSIS REPORT")
        print(f"{'='*80}")
        print(f"SMILES: {results['smiles']}")
        print(f"{'='*80}")
        
        # Group properties by category
        categories = {
            'Physical Properties': ['molecular_weight', 'solubility', 'lipophilicity', 'polar_surface_area', 'vapor_pressure'],
            'Structural Properties': ['hydrogen_bond_donors', 'hydrogen_bond_acceptors', 'rotatable_bonds', 'fluorine_count', 'fluorine_content'],
            'Drug-like Properties': ['drug_likeness', 'membrane_permeability', 'metabolic_stability'],
            'Environmental Properties': ['bioaccumulation_potential', 'environmental_persistence', 'toxicity_risk', 'water_solubility_class']
        }
        
        for category, prop_names in categories.items():
            print(f"\n{category.upper()}:")
            print("-" * 50)
            
            for prop_name in prop_names:
                if prop_name in results['properties']:
                    prop_data = results['properties'][prop_name]
                    if prop_data['value'] is None:
                        continue
                    
                    print(f"\n{prop_name.replace('_', ' ').title()}:")
                    print(f"  Description: {prop_data['description']}")
                    
                    if prop_name == 'water_solubility_class':
                        print(f"  Value: {prop_data['value']}")
                    else:
                        print(f"  Value: {prop_data['value']:.4f} {prop_data['unit']}")
                    
                    print(f"  Type: {prop_data['type']}")
                    
                    # Add interpretation
                    interpretation = self._get_interpretation(prop_name, prop_data['value'])
                    if interpretation:
                        print(f"  Interpretation: {interpretation}")
        
        print(f"\n{'='*80}")
    
    def _get_interpretation(self, prop_name: str, value) -> str:
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
        
        elif prop_name == 'fluorine_count':
            if value <= 5:
                return interpretations['low']
            elif value <= 15:
                return interpretations['moderate']
            else:
                return interpretations['high']
        
        elif prop_name == 'drug_likeness':
            if value >= 4:
                return interpretations['good']
            elif value >= 2:
                return interpretations['moderate']
            else:
                return interpretations['poor']
        
        elif prop_name in ['bioaccumulation_potential', 'environmental_persistence', 'toxicity_risk', 'membrane_permeability', 'metabolic_stability']:
            if value <= 2:
                return interpretations['low']
            elif value <= 4:
                return interpretations['moderate']
            else:
                return interpretations['high']
        
        elif prop_name == 'water_solubility_class':
            return interpretations.get(value, "Unknown")
        
        elif prop_name == 'vapor_pressure':
            if value > 1000:
                return interpretations['high']
            elif value > 10:
                return interpretations['moderate']
            else:
                return interpretations['low']
        
        return None


def main():
    """Demo the enhanced PFAS predictor."""
    print("Enhanced Fast PFAS Property Predictor Demo")
    print("=" * 60)
    
    # Initialize predictor
    predictor = EnhancedPFASPredictor()
    
    # Test with PFAS compounds
    pfas_compounds = {
        "PFOA": "C(=O)(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O",
        "PFOS": "C(C(C(C(C(F)(F)S(=O)(=O)O)(F)F)(F)F)(F)F)(C(C(C(F)(F)F)(F)F)(F)F)(F)F",
        "PFHxA": "C(=O)(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)O",
        "PFBS": "C(C(C(F)(F)S(=O)(=O)O)(F)F)(C(F)(F)F)(F)F",
        "GenX": "C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)OC1=CC=C(C=C1)C(=O)O",
    }
    
    print(f"\nAvailable properties: {len(predictor.available_properties)} properties")
    print(f"Property categories: Physical, Structural, Drug-like, Environmental")
    
    # Predict all properties for each compound
    for name, smiles in pfas_compounds.items():
        print(f"\n{'='*50}")
        print(f"Analyzing {name}...")
        print(f"{'='*50}")
        
        # Get predictions
        results = predictor.predict_all_properties(smiles)
        
        # Print detailed report
        predictor.print_compound_report(results)
    
    print(f"\nDemo completed successfully! 🎉")


if __name__ == "__main__":
    main() 
"""
Accurate PFAS Property Predictor (modular version)

A more accurate property prediction system for PFAS compounds using
advanced RDKit calculations and better models for structural properties.

Author: PFAS AI Project
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import warnings
import os
import joblib
warnings.filterwarnings('ignore')

import deepchem as dc
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit import DataStructs
from sklearn.ensemble import RandomForestRegressor


class AccuratePFASPredictor:
    """
    Accurate property predictor for PFAS compounds with precise structural calculations.
    
    Uses advanced RDKit methods for more accurate property predictions.
    """
    
    def __init__(self):
        """Initialize the accurate predictor."""
        self.models = {}
        self.property_info = {}
        self.model_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        # Define comprehensive property set with more accurate descriptions
        self.available_properties = {
            'molecular_weight': {
                'description': 'Exact molecular weight (monoisotopic)',
                'unit': 'g/mol',
                'type': 'calculated',
                'method': 'rdMolDescriptors.CalcExactMolWt'
            },
            'formula': {
                'description': 'Molecular formula',
                'unit': 'formula',
                'type': 'calculated',
                'method': 'rdMolDescriptors.CalcMolFormula'
            },
            'solubility': {
                'description': 'Aqueous solubility (logS)',
                'unit': 'log(mol/L)',
                'type': 'predicted',
                'method': 'ML model + RDKit estimation'
            },
            'logp': {
                'description': 'Octanol-water partition coefficient (Wildman-Crippen)',
                'unit': 'logP',
                'type': 'calculated',
                'method': 'rdMolDescriptors.CalcCrippenDescriptors'
            },
            'logd': {
                'description': 'Distribution coefficient at pH 7.4',
                'unit': 'logD',
                'type': 'calculated',
                'method': 'rdMolDescriptors.CalcCrippenDescriptors'
            },
            'tpsa': {
                'description': 'Topological polar surface area (Ertl)',
                'unit': 'Å²',
                'type': 'calculated',
                'method': 'rdMolDescriptors.CalcTPSA'
            },
            'hbd': {
                'description': 'Number of hydrogen bond donors (Lipinski)',
                'unit': 'count',
                'type': 'calculated',
                'method': 'rdMolDescriptors.CalcNumHBD'
            },
            'hba': {
                'description': 'Number of hydrogen bond acceptors (Lipinski)',
                'unit': 'count',
                'type': 'calculated',
                'method': 'rdMolDescriptors.CalcNumHBA'
            },
            'rotatable_bonds': {
                'description': 'Number of rotatable bonds (strict definition)',
                'unit': 'count',
                'type': 'calculated',
                'method': 'rdMolDescriptors.CalcNumRotatableBonds'
            },
            'aromatic_rings': {
                'description': 'Number of aromatic rings',
                'unit': 'count',
                'type': 'calculated',
                'method': 'rdMolDescriptors.CalcNumAromaticRings'
            },
            'saturated_rings': {
                'description': 'Number of saturated rings',
                'unit': 'count',
                'type': 'calculated',
                'method': 'rdMolDescriptors.CalcNumSaturatedRings'
            },
            'heteroatoms': {
                'description': 'Number of heteroatoms',
                'unit': 'count',
                'type': 'calculated',
                'method': 'rdMolDescriptors.CalcNumHeteroatoms'
            },
            'fluorine_count': {
                'description': 'Number of fluorine atoms',
                'unit': 'count',
                'type': 'calculated',
                'method': 'Atom counting'
            },
            'fluorine_percentage': {
                'description': 'Percentage of fluorine atoms',
                'unit': '%',
                'type': 'calculated',
                'method': 'Atom counting'
            },
            'carbon_count': {
                'description': 'Number of carbon atoms',
                'unit': 'count',
                'type': 'calculated',
                'method': 'Atom counting'
            },
            'oxygen_count': {
                'description': 'Number of oxygen atoms',
                'unit': 'count',
                'type': 'calculated',
                'method': 'Atom counting'
            },
            'sulfur_count': {
                'description': 'Number of sulfur atoms',
                'unit': 'count',
                'type': 'calculated',
                'method': 'Atom counting'
            },
            'nitrogen_count': {
                'description': 'Number of nitrogen atoms',
                'unit': 'count',
                'type': 'calculated',
                'method': 'Atom counting'
            },
            'hydrogen_count': {
                'description': 'Number of hydrogen atoms',
                'unit': 'count',
                'type': 'calculated',
                'method': 'rdMolDescriptors.CalcNumHeteroatoms'
            },
            'amide_bonds': {
                'description': 'Number of amide bonds',
                'unit': 'count',
                'type': 'calculated',
                'method': 'rdMolDescriptors.CalcNumAmideBonds'
            },
            'ester_bonds': {
                'description': 'Number of ester bonds',
                'unit': 'count',
                'type': 'calculated',
                'method': 'rdMolDescriptors.CalcNumEsterBonds'
            },
            'ether_bonds': {
                'description': 'Number of ether bonds',
                'unit': 'count',
                'type': 'calculated',
                'method': 'rdMolDescriptors.CalcNumEtherBonds'
            },
            'sulfonamides': {
                'description': 'Number of sulfonamide groups',
                'unit': 'count',
                'type': 'calculated',
                'method': 'rdMolDescriptors.CalcNumSulfonamides'
            },
            'sulfones': {
                'description': 'Number of sulfone groups',
                'unit': 'count',
                'type': 'calculated',
                'method': 'rdMolDescriptors.CalcNumSulfones'
            },
            'fraction_csp3': {
                'description': 'Fraction of sp3 hybridized carbons',
                'unit': 'fraction',
                'type': 'calculated',
                'method': 'rdMolDescriptors.CalcFractionCsp3'
            },
            'spiro_atoms': {
                'description': 'Number of spiro atoms',
                'unit': 'count',
                'type': 'calculated',
                'method': 'rdMolDescriptors.CalcNumSpiroAtoms'
            },
            'bridgehead_atoms': {
                'description': 'Number of bridgehead atoms',
                'unit': 'count',
                'type': 'calculated',
                'method': 'rdMolDescriptors.CalcNumBridgeheadAtoms'
            },
            'drug_likeness': {
                'description': 'Lipinski drug-likeness score',
                'unit': 'score',
                'type': 'calculated',
                'method': 'Lipinski rules'
            },
            'bioaccumulation_potential': {
                'description': 'Bioaccumulation potential score',
                'unit': 'score',
                'type': 'calculated',
                'method': 'PFAS-specific scoring'
            },
            'environmental_persistence': {
                'description': 'Environmental persistence score',
                'unit': 'score',
                'type': 'calculated',
                'method': 'PFAS-specific scoring'
            },
            'toxicity_risk': {
                'description': 'Toxicity risk assessment',
                'unit': 'score',
                'type': 'calculated',
                'method': 'PFAS-specific scoring'
            }
        }
        
        # Load solubility model
        self._load_solubility_model()
    
    def _get_model_path(self, prop_name):
        return os.path.join(self.model_dir, f"{prop_name}_model.joblib")
    
    def _load_solubility_model(self):
        """Load a simple solubility prediction model."""
        model_path = self._get_model_path('solubility')
        try:
            if os.path.exists(model_path):
                print("Loading solubility prediction model from disk...")
                model = joblib.load(model_path)
                # Load dataset to get transformers
                tasks, datasets, transformers = dc.molnet.load_delaney(
                    featurizer='MorganGenerator', split='random'
                )
                self.models['solubility'] = {
                    'model': model,
                    'transformer': transformers[0] if transformers else None
                }
                print(f"✓ Solubility model loaded from disk")
                return
            print("Training solubility prediction model...")
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
            # Save model to disk
            joblib.dump(model, model_path)
            self.models['solubility'] = {
                'model': model,
                'transformer': transformers[0] if transformers else None
            }
            print(f"✓ Solubility model trained and saved to disk")
            print(f"  - Training samples: {len(train_dataset)}")
        except Exception as e:
            print(f"⚠️  Solubility model loading failed: {e}")
            print("  Will use RDKit-based estimation instead")
    
    def extract_accurate_features(self, smiles: str) -> Dict:
        """
        Extract accurate molecular features using advanced RDKit methods.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Dictionary containing accurate molecular features
        """
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            return None
        
        features = {}
        
        # Exact molecular weight (monoisotopic)
        try:
            features['molecular_weight'] = rdMolDescriptors.CalcExactMolWt(mol)
        except:
            features['molecular_weight'] = Chem.Descriptors.MolWt(mol)
        
        # Molecular formula
        try:
            features['formula'] = rdMolDescriptors.CalcMolFormula(mol)
        except:
            features['formula'] = "Unknown"
        
        # Crippen descriptors (more accurate logP and logD)
        try:
            crippen_descriptors = rdMolDescriptors.CalcCrippenDescriptors(mol)
            features['logp'] = crippen_descriptors[0]  # LogP
            features['logd'] = crippen_descriptors[1]  # LogD
        except:
            features['logp'] = Chem.Descriptors.MolLogP(mol)
            features['logd'] = Chem.Descriptors.MolLogP(mol)  # Fallback
        
        # TPSA (Ertl method)
        try:
            features['tpsa'] = rdMolDescriptors.CalcTPSA(mol)
        except:
            features['tpsa'] = Chem.Descriptors.TPSA(mol)
        
        # Hydrogen bonding (Lipinski method)
        try:
            features['hbd'] = rdMolDescriptors.CalcNumHBD(mol)
        except:
            features['hbd'] = Chem.Descriptors.NumHDonors(mol)
        
        try:
            features['hba'] = rdMolDescriptors.CalcNumHBA(mol)
        except:
            features['hba'] = Chem.Descriptors.NumHAcceptors(mol)
        
        # Rotatable bonds (strict definition)
        try:
            features['rotatable_bonds'] = rdMolDescriptors.CalcNumRotatableBonds(mol)
        except:
            features['rotatable_bonds'] = Chem.Descriptors.NumRotatableBonds(mol)
        
        # Ring systems
        try:
            features['aromatic_rings'] = rdMolDescriptors.CalcNumAromaticRings(mol)
        except:
            features['aromatic_rings'] = Chem.Descriptors.NumAromaticRings(mol)
        
        try:
            features['saturated_rings'] = rdMolDescriptors.CalcNumSaturatedRings(mol)
        except:
            features['saturated_rings'] = Chem.Descriptors.NumSaturatedRings(mol)
        
        # Heteroatoms
        try:
            features['heteroatoms'] = rdMolDescriptors.CalcNumHeteroatoms(mol)
        except:
            features['heteroatoms'] = Chem.Descriptors.NumHeteroatoms(mol)
        
        # Hydrogen count
        try:
            features['hydrogen_count'] = rdMolDescriptors.CalcNumHeteroatoms(mol)
        except:
            features['hydrogen_count'] = 0
        
        # Bond types
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
        
        # Fraction of sp3 carbons
        try:
            features['fraction_csp3'] = rdMolDescriptors.CalcFractionCsp3(mol)
        except:
            features['fraction_csp3'] = Chem.Descriptors.FractionCSP3(mol)
        
        # Spiro and bridgehead atoms
        try:
            features['spiro_atoms'] = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        except:
            features['spiro_atoms'] = Chem.Descriptors.NumSpiroAtoms(mol)
        
        try:
            features['bridgehead_atoms'] = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        except:
            features['bridgehead_atoms'] = Chem.Descriptors.NumBridgeheadAtoms(mol)
        
        # Atom counting (more accurate)
        fluorine_count = 0
        carbon_count = 0
        oxygen_count = 0
        sulfur_count = 0
        nitrogen_count = 0
        hydrogen_count = 0
        
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
            elif symbol == 'N':
                nitrogen_count += 1
            elif symbol == 'H':
                hydrogen_count += 1
        
        total_atoms = mol.GetNumAtoms()
        fluorine_percentage = (fluorine_count / total_atoms * 100) if total_atoms > 0 else 0
        
        features['fluorine_count'] = fluorine_count
        features['fluorine_percentage'] = fluorine_percentage
        features['carbon_count'] = carbon_count
        features['oxygen_count'] = oxygen_count
        features['sulfur_count'] = sulfur_count
        features['nitrogen_count'] = nitrogen_count
        features['hydrogen_count'] = hydrogen_count
        
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
        
        # Enhanced solubility estimation using Crippen descriptors
        try:
            crippen_descriptors = rdMolDescriptors.CalcCrippenDescriptors(mol)
            logp = crippen_descriptors[0]
            mw = rdMolDescriptors.CalcExactMolWt(mol)
            tpsa = rdMolDescriptors.CalcTPSA(mol)
        except:
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
        
        # Extract accurate features
        features = self.extract_accurate_features(smiles)
        
        if features is None:
            print("❌ Invalid SMILES string")
            return None
        
        # Calculate all properties
        properties = {}
        
        # Basic properties
        properties['molecular_weight'] = {
            'value': features['molecular_weight'],
            'description': self.available_properties['molecular_weight']['description'],
            'unit': self.available_properties['molecular_weight']['unit'],
            'type': 'calculated'
        }
        
        properties['formula'] = {
            'value': features['formula'],
            'description': self.available_properties['formula']['description'],
            'unit': self.available_properties['formula']['unit'],
            'type': 'calculated'
        }
        
        properties['solubility'] = {
            'value': self.predict_solubility(smiles),
            'description': self.available_properties['solubility']['description'],
            'unit': self.available_properties['solubility']['unit'],
            'type': 'predicted'
        }
        
        properties['logp'] = {
            'value': features['logp'],
            'description': self.available_properties['logp']['description'],
            'unit': self.available_properties['logp']['unit'],
            'type': 'calculated'
        }
        
        properties['logd'] = {
            'value': features['logd'],
            'description': self.available_properties['logd']['description'],
            'unit': self.available_properties['logd']['unit'],
            'type': 'calculated'
        }
        
        properties['tpsa'] = {
            'value': features['tpsa'],
            'description': self.available_properties['tpsa']['description'],
            'unit': self.available_properties['tpsa']['unit'],
            'type': 'calculated'
        }
        
        properties['hbd'] = {
            'value': features['hbd'],
            'description': self.available_properties['hbd']['description'],
            'unit': self.available_properties['hbd']['unit'],
            'type': 'calculated'
        }
        
        properties['hba'] = {
            'value': features['hba'],
            'description': self.available_properties['hba']['description'],
            'unit': self.available_properties['hba']['unit'],
            'type': 'calculated'
        }
        
        properties['rotatable_bonds'] = {
            'value': features['rotatable_bonds'],
            'description': self.available_properties['rotatable_bonds']['description'],
            'unit': self.available_properties['rotatable_bonds']['unit'],
            'type': 'calculated'
        }
        
        properties['aromatic_rings'] = {
            'value': features['aromatic_rings'],
            'description': self.available_properties['aromatic_rings']['description'],
            'unit': self.available_properties['aromatic_rings']['unit'],
            'type': 'calculated'
        }
        
        properties['saturated_rings'] = {
            'value': features['saturated_rings'],
            'description': self.available_properties['saturated_rings']['description'],
            'unit': self.available_properties['saturated_rings']['unit'],
            'type': 'calculated'
        }
        
        properties['heteroatoms'] = {
            'value': features['heteroatoms'],
            'description': self.available_properties['heteroatoms']['description'],
            'unit': self.available_properties['heteroatoms']['unit'],
            'type': 'calculated'
        }
        
        properties['fluorine_count'] = {
            'value': features['fluorine_count'],
            'description': self.available_properties['fluorine_count']['description'],
            'unit': self.available_properties['fluorine_count']['unit'],
            'type': 'calculated'
        }
        
        properties['fluorine_percentage'] = {
            'value': features['fluorine_percentage'],
            'description': self.available_properties['fluorine_percentage']['description'],
            'unit': self.available_properties['fluorine_percentage']['unit'],
            'type': 'calculated'
        }
        
        # Atom counts
        for atom_type in ['carbon_count', 'oxygen_count', 'sulfur_count', 'nitrogen_count', 'hydrogen_count']:
            properties[atom_type] = {
                'value': features[atom_type],
                'description': self.available_properties[atom_type]['description'],
                'unit': self.available_properties[atom_type]['unit'],
                'type': 'calculated'
            }
        
        # Bond types
        for bond_type in ['amide_bonds', 'ester_bonds', 'ether_bonds', 'sulfonamides', 'sulfones']:
            properties[bond_type] = {
                'value': features[bond_type],
                'description': self.available_properties[bond_type]['description'],
                'unit': self.available_properties[bond_type]['unit'],
                'type': 'calculated'
            }
        
        # Advanced structural properties
        properties['fraction_csp3'] = {
            'value': features['fraction_csp3'],
            'description': self.available_properties['fraction_csp3']['description'],
            'unit': self.available_properties['fraction_csp3']['unit'],
            'type': 'calculated'
        }
        
        properties['spiro_atoms'] = {
            'value': features['spiro_atoms'],
            'description': self.available_properties['spiro_atoms']['description'],
            'unit': self.available_properties['spiro_atoms']['unit'],
            'type': 'calculated'
        }
        
        properties['bridgehead_atoms'] = {
            'value': features['bridgehead_atoms'],
            'description': self.available_properties['bridgehead_atoms']['description'],
            'unit': self.available_properties['bridgehead_atoms']['unit'],
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
        print(f"ACCURATE PFAS COMPOUND ANALYSIS REPORT")
        print(f"{'='*80}")
        print(f"SMILES: {results['smiles']}")
        print(f"{'='*80}")
        
        # Group properties by category
        categories = {
            'Molecular Identity': ['molecular_weight', 'formula'],
            'Physical Properties': ['solubility', 'logp', 'logd', 'tpsa'],
            'Structural Properties': ['hbd', 'hba', 'rotatable_bonds', 'aromatic_rings', 'saturated_rings', 'heteroatoms'],
            'Atom Composition': ['carbon_count', 'hydrogen_count', 'nitrogen_count', 'oxygen_count', 'sulfur_count', 'fluorine_count', 'fluorine_percentage'],
            'Bond Types': ['amide_bonds', 'ester_bonds', 'ether_bonds', 'sulfonamides', 'sulfones'],
            'Advanced Structural': ['fraction_csp3', 'spiro_atoms', 'bridgehead_atoms'],
            'Assessment Properties': ['drug_likeness', 'bioaccumulation_potential', 'environmental_persistence', 'toxicity_risk']
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
                    
                    if prop_name == 'formula':
                        print(f"  Value: {prop_data['value']}")
                    else:
                        print(f"  Value: {prop_data['value']:.4f} {prop_data['unit']}")
                    
                    print(f"  Type: {prop_data['type']}")
                    print(f"  Method: {self.available_properties[prop_name]['method']}")
        
        print(f"\n{'='*80}")

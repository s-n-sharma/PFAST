"""
Simple PFAS Analyzer

A fast and simple property analyzer for PFAS compounds using only RDKit.
No model training required - provides immediate results.

Author: PFAS AI Project
Date: 2024
"""

import numpy as np
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Core imports
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem


class SimplePFASAnalyzer:
    """
    Simple PFAS compound analyzer using RDKit descriptors.
    Provides immediate property calculations without model training.
    """
    
    def __init__(self):
        """Initialize the analyzer."""
        self.property_info = {
            'molecular_weight': {
                'description': 'Molecular weight',
                'unit': 'g/mol',
                'interpretation': {
                    'low': '< 300 (Small molecule)',
                    'moderate': '300-500 (Medium molecule)',
                    'high': '> 500 (Large molecule)'
                }
            },
            'logp': {
                'description': 'Octanol-water partition coefficient',
                'unit': 'logP',
                'interpretation': {
                    'low': '< 2 (Good for drug-like properties)',
                    'moderate': '2-5 (Moderate lipophilicity)',
                    'high': '> 5 (High lipophilicity, bioavailability issues)'
                }
            },
            'tpsa': {
                'description': 'Topological polar surface area',
                'unit': 'Å²',
                'interpretation': {
                    'low': '< 90 (Good membrane permeability)',
                    'moderate': '90-140 (Moderate permeability)',
                    'high': '> 140 (Poor membrane permeability)'
                }
            },
            'hbd': {
                'description': 'Number of hydrogen bond donors',
                'unit': 'count',
                'interpretation': {
                    'low': '0-2 (Good for drug-like properties)',
                    'moderate': '3-5 (Moderate)',
                    'high': '> 5 (May have bioavailability issues)'
                }
            },
            'hba': {
                'description': 'Number of hydrogen bond acceptors',
                'unit': 'count',
                'interpretation': {
                    'low': '0-5 (Good for drug-like properties)',
                    'moderate': '6-10 (Moderate)',
                    'high': '> 10 (May have bioavailability issues)'
                }
            },
            'rotatable_bonds': {
                'description': 'Number of rotatable bonds',
                'unit': 'count',
                'interpretation': {
                    'low': '0-3 (Rigid molecule)',
                    'moderate': '4-7 (Moderate flexibility)',
                    'high': '> 7 (Very flexible)'
                }
            },
            'aromatic_rings': {
                'description': 'Number of aromatic rings',
                'unit': 'count',
                'interpretation': {
                    'low': '0-1 (Few aromatic rings)',
                    'moderate': '2-3 (Moderate aromaticity)',
                    'high': '> 3 (Highly aromatic)'
                }
            },
            'fluorine_count': {
                'description': 'Number of fluorine atoms',
                'unit': 'count',
                'interpretation': {
                    'low': '0-5 (Low fluorination)',
                    'moderate': '6-15 (Moderate fluorination)',
                    'high': '> 15 (Highly fluorinated)'
                }
            },
            'fluorine_percentage': {
                'description': 'Percentage of fluorine atoms',
                'unit': '%',
                'interpretation': {
                    'low': '< 30% (Low fluorination)',
                    'moderate': '30-60% (Moderate fluorination)',
                    'high': '> 60% (Highly fluorinated)'
                }
            },
            'estimated_solubility': {
                'description': 'Estimated aqueous solubility (logS)',
                'unit': 'log(mol/L)',
                'interpretation': {
                    'high': '> -2 (Highly soluble)',
                    'moderate': '-4 to -2 (Moderately soluble)',
                    'low': '< -4 (Poorly soluble)'
                }
            },
            'drug_likeness': {
                'description': 'Lipinski drug-likeness score',
                'unit': 'score',
                'interpretation': {
                    'good': '≥ 4 (Good drug-like properties)',
                    'moderate': '2-3 (Moderate drug-like properties)',
                    'poor': '0-1 (Poor drug-like properties)'
                }
            }
        }
    
    def analyze_compound(self, smiles: str) -> Optional[Dict]:
        """
        Analyze a PFAS compound and return all calculated properties.
        
        Args:
            smiles: SMILES string of the compound
            
        Returns:
            Dictionary containing all calculated properties
        """
        print(f"Analyzing compound: {smiles}")
        
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            print("❌ Invalid SMILES string")
            return None
        
        # Calculate basic properties
        properties = {}
        
        # Molecular weight
        properties['molecular_weight'] = {
            'value': Chem.Descriptors.MolWt(mol),
            'description': self.property_info['molecular_weight']['description'],
            'unit': self.property_info['molecular_weight']['unit']
        }
        
        # LogP
        properties['logp'] = {
            'value': Chem.Descriptors.MolLogP(mol),
            'description': self.property_info['logp']['description'],
            'unit': self.property_info['logp']['unit']
        }
        
        # Topological polar surface area
        properties['tpsa'] = {
            'value': Chem.Descriptors.TPSA(mol),
            'description': self.property_info['tpsa']['description'],
            'unit': self.property_info['tpsa']['unit']
        }
        
        # Hydrogen bond donors
        properties['hbd'] = {
            'value': Chem.Descriptors.NumHDonors(mol),
            'description': self.property_info['hbd']['description'],
            'unit': self.property_info['hbd']['unit']
        }
        
        # Hydrogen bond acceptors
        properties['hba'] = {
            'value': Chem.Descriptors.NumHAcceptors(mol),
            'description': self.property_info['hba']['description'],
            'unit': self.property_info['hba']['unit']
        }
        
        # Rotatable bonds
        properties['rotatable_bonds'] = {
            'value': Chem.Descriptors.NumRotatableBonds(mol),
            'description': self.property_info['rotatable_bonds']['description'],
            'unit': self.property_info['rotatable_bonds']['unit']
        }
        
        # Aromatic rings
        properties['aromatic_rings'] = {
            'value': Chem.Descriptors.NumAromaticRings(mol),
            'description': self.property_info['aromatic_rings']['description'],
            'unit': self.property_info['aromatic_rings']['unit']
        }
        
        # Fluorine analysis
        fluorine_count = 0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'F':
                fluorine_count += 1
        
        total_atoms = mol.GetNumAtoms()
        fluorine_percentage = (fluorine_count / total_atoms * 100) if total_atoms > 0 else 0
        
        properties['fluorine_count'] = {
            'value': fluorine_count,
            'description': self.property_info['fluorine_count']['description'],
            'unit': self.property_info['fluorine_count']['unit']
        }
        
        properties['fluorine_percentage'] = {
            'value': fluorine_percentage,
            'description': self.property_info['fluorine_percentage']['description'],
            'unit': self.property_info['fluorine_percentage']['unit']
        }
        
        # Estimated solubility (simplified)
        estimated_logs = -properties['logp']['value'] - 0.01 * properties['molecular_weight']['value'] + 1.5
        properties['estimated_solubility'] = {
            'value': estimated_logs,
            'description': self.property_info['estimated_solubility']['description'],
            'unit': self.property_info['estimated_solubility']['unit']
        }
        
        # Drug-likeness score (Lipinski rules)
        drug_score = 0
        if properties['molecular_weight']['value'] <= 500:
            drug_score += 1
        if properties['logp']['value'] <= 5:
            drug_score += 1
        if properties['hbd']['value'] <= 5:
            drug_score += 1
        if properties['hba']['value'] <= 10:
            drug_score += 1
        
        properties['drug_likeness'] = {
            'value': drug_score,
            'description': self.property_info['drug_likeness']['description'],
            'unit': self.property_info['drug_likeness']['unit']
        }
        
        return {
            'smiles': smiles,
            'properties': properties
        }
    
    def print_analysis_report(self, results: Dict):
        """Print a formatted analysis report."""
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
            
            # Add interpretation
            interpretation = self._get_interpretation(prop_name, prop_data['value'])
            if interpretation:
                print(f"  Interpretation: {interpretation}")
        
        print(f"\n{'='*70}")
    
    def _get_interpretation(self, prop_name: str, value: float) -> str:
        """Get interpretation for a property value."""
        if prop_name not in self.property_info:
            return None
        
        interpretations = self.property_info[prop_name]['interpretation']
        
        if prop_name == 'molecular_weight':
            if value < 300:
                return interpretations['low']
            elif value < 500:
                return interpretations['moderate']
            else:
                return interpretations['high']
        
        elif prop_name == 'logp':
            if value < 2:
                return interpretations['low']
            elif value < 5:
                return interpretations['moderate']
            else:
                return interpretations['high']
        
        elif prop_name == 'tpsa':
            if value < 90:
                return interpretations['low']
            elif value < 140:
                return interpretations['moderate']
            else:
                return interpretations['high']
        
        elif prop_name == 'hbd':
            if value <= 2:
                return interpretations['low']
            elif value <= 5:
                return interpretations['moderate']
            else:
                return interpretations['high']
        
        elif prop_name == 'hba':
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
        
        elif prop_name == 'aromatic_rings':
            if value <= 1:
                return interpretations['low']
            elif value <= 3:
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
        
        elif prop_name == 'fluorine_percentage':
            if value < 30:
                return interpretations['low']
            elif value < 60:
                return interpretations['moderate']
            else:
                return interpretations['high']
        
        elif prop_name == 'estimated_solubility':
            if value > -2:
                return interpretations['high']
            elif value > -4:
                return interpretations['moderate']
            else:
                return interpretations['low']
        
        elif prop_name == 'drug_likeness':
            if value >= 4:
                return interpretations['good']
            elif value >= 2:
                return interpretations['moderate']
            else:
                return interpretations['poor']
        
        return None


def main():
    """Demo the simple PFAS analyzer."""
    print("Simple PFAS Analyzer Demo")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = SimplePFASAnalyzer()
    
    # Test with PFAS compounds
    pfas_compounds = {
        "PFOA": "C(=O)(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O",
        "PFOS": "C(C(C(C(C(F)(F)S(=O)(=O)O)(F)F)(F)F)(F)F)(C(C(C(F)(F)F)(F)F)(F)F)(F)F",
        "PFHxA": "C(=O)(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)O",
        "PFBS": "C(C(C(F)(F)S(=O)(=O)O)(F)F)(C(F)(F)F)(F)F",
    }
    
    print(f"\nAvailable properties: {list(analyzer.property_info.keys())}")
    
    # Analyze each compound
    for name, smiles in pfas_compounds.items():
        print(f"\n{'='*40}")
        print(f"Analyzing {name}...")
        print(f"{'='*40}")
        
        # Get analysis
        results = analyzer.analyze_compound(smiles)
        
        # Print detailed report
        analyzer.print_analysis_report(results)
    
    print(f"\nDemo completed successfully! 🎉")


if __name__ == "__main__":
    main() 
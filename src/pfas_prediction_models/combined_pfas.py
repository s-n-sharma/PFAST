"""
Combined PFAS Property Predictor

Combines AccuratePFASPredictor, MoleculeNetPFASPredictor, and numerical counting.
Intelligently selects the best prediction method for each property.

Author: PFAS AI Project
Date: 2024
"""

try:
    from .accurate_pfas import AccuratePFASPredictor
    from .moleculenet_pfas import MoleculeNetPFASPredictor
    from .utils import is_valid_smiles
except ImportError:
    # Fallback for when running directly
    from accurate_pfas import AccuratePFASPredictor
    from moleculenet_pfas import MoleculeNetPFASPredictor
    from utils import is_valid_smiles

from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class CombinedPFASPredictor:
    """
    Combined PFAS property predictor that intelligently selects the best method.
    
    Property selection strategy:
    - MoleculeNet: solubility, lipophilicity, bbbp, hiv (when available)
    - RDKit: molecular_weight, formula, tpsa, hbd, hba, rotatable_bonds, etc.
    - Combined: advanced properties (drug_likeness, bioaccumulation, etc.)
    """
    
    def __init__(self):
        self.accurate = AccuratePFASPredictor()
        self.moleculenet = MoleculeNetPFASPredictor()
        
        # Define property mapping and priority
        self.property_mapping = {
            # MoleculeNet properties (higher priority when available)
            'solubility': {
                'primary': 'moleculenet',
                'fallback': 'accurate',
                'description': 'Aqueous solubility (logS)',
                'unit': 'log(mol/L)'
            },
            'lipophilicity': {
                'primary': 'moleculenet', 
                'fallback': 'accurate',
                'description': 'Octanol-water partition coefficient',
                'unit': 'logP'
            },
            'bbbp': {
                'primary': 'moleculenet',
                'fallback': None,
                'description': 'Blood-brain barrier penetration',
                'unit': 'probability'
            },
            'hiv': {
                'primary': 'moleculenet',
                'fallback': None,
                'description': 'HIV replication inhibition',
                'unit': 'probability'
            },
            
            # RDKit properties (accurate predictor)
            'molecular_weight': {
                'primary': 'accurate',
                'fallback': None,
                'description': 'Exact molecular weight (monoisotopic)',
                'unit': 'g/mol'
            },
            'formula': {
                'primary': 'accurate',
                'fallback': None,
                'description': 'Molecular formula',
                'unit': 'formula'
            },
            'logp': {
                'primary': 'accurate',
                'fallback': None,
                'description': 'Octanol-water partition coefficient (Wildman-Crippen)',
                'unit': 'logP'
            },
            'logd': {
                'primary': 'accurate',
                'fallback': None,
                'description': 'Distribution coefficient at pH 7.4',
                'unit': 'logD'
            },
            'tpsa': {
                'primary': 'accurate',
                'fallback': None,
                'description': 'Topological polar surface area (Ertl)',
                'unit': 'Å²'
            },
            'hbd': {
                'primary': 'accurate',
                'fallback': None,
                'description': 'Number of hydrogen bond donors (Lipinski)',
                'unit': 'count'
            },
            'hba': {
                'primary': 'accurate',
                'fallback': None,
                'description': 'Number of hydrogen bond acceptors (Lipinski)',
                'unit': 'count'
            },
            'rotatable_bonds': {
                'primary': 'accurate',
                'fallback': None,
                'description': 'Number of rotatable bonds (strict definition)',
                'unit': 'count'
            },
            'aromatic_rings': {
                'primary': 'accurate',
                'fallback': None,
                'description': 'Number of aromatic rings',
                'unit': 'count'
            },
            'saturated_rings': {
                'primary': 'accurate',
                'fallback': None,
                'description': 'Number of saturated rings',
                'unit': 'count'
            },
            'heteroatoms': {
                'primary': 'accurate',
                'fallback': None,
                'description': 'Number of heteroatoms',
                'unit': 'count'
            },
            'fluorine_count': {
                'primary': 'accurate',
                'fallback': None,
                'description': 'Number of fluorine atoms',
                'unit': 'count'
            },
            'fluorine_percentage': {
                'primary': 'accurate',
                'fallback': None,
                'description': 'Percentage of fluorine atoms',
                'unit': '%'
            },
            'carbon_count': {
                'primary': 'accurate',
                'fallback': None,
                'description': 'Number of carbon atoms',
                'unit': 'count'
            },
            'oxygen_count': {
                'primary': 'accurate',
                'fallback': None,
                'description': 'Number of oxygen atoms',
                'unit': 'count'
            },
            'sulfur_count': {
                'primary': 'accurate',
                'fallback': None,
                'description': 'Number of sulfur atoms',
                'unit': 'count'
            },
            'nitrogen_count': {
                'primary': 'accurate',
                'fallback': None,
                'description': 'Number of nitrogen atoms',
                'unit': 'count'
            },
            'hydrogen_count': {
                'primary': 'accurate',
                'fallback': None,
                'description': 'Number of hydrogen atoms',
                'unit': 'count'
            },
            'amide_bonds': {
                'primary': 'accurate',
                'fallback': None,
                'description': 'Number of amide bonds',
                'unit': 'count'
            },
            'ester_bonds': {
                'primary': 'accurate',
                'fallback': None,
                'description': 'Number of ester bonds',
                'unit': 'count'
            },
            'ether_bonds': {
                'primary': 'accurate',
                'fallback': None,
                'description': 'Number of ether bonds',
                'unit': 'count'
            },
            'sulfonamides': {
                'primary': 'accurate',
                'fallback': None,
                'description': 'Number of sulfonamide groups',
                'unit': 'count'
            },
            'sulfones': {
                'primary': 'accurate',
                'fallback': None,
                'description': 'Number of sulfone groups',
                'unit': 'count'
            },
            'fraction_csp3': {
                'primary': 'accurate',
                'fallback': None,
                'description': 'Fraction of sp3 hybridized carbons',
                'unit': 'fraction'
            },
            'spiro_atoms': {
                'primary': 'accurate',
                'fallback': None,
                'description': 'Number of spiro atoms',
                'unit': 'count'
            },
            'bridgehead_atoms': {
                'primary': 'accurate',
                'fallback': None,
                'description': 'Number of bridgehead atoms',
                'unit': 'count'
            },
            
            # Combined/Advanced properties
            'drug_likeness': {
                'primary': 'accurate',
                'fallback': None,
                'description': 'Lipinski drug-likeness score',
                'unit': 'score'
            },
            'bioaccumulation_potential': {
                'primary': 'accurate',
                'fallback': None,
                'description': 'Bioaccumulation potential score',
                'unit': 'score'
            },
            'environmental_persistence': {
                'primary': 'accurate',
                'fallback': None,
                'description': 'Environmental persistence score',
                'unit': 'score'
            },
            'toxicity_risk': {
                'primary': 'accurate',
                'fallback': None,
                'description': 'Toxicity risk assessment',
                'unit': 'score'
            }
        }
    
    def predict_all(self, smiles: str) -> Dict:
        """
        Predict all properties using the best available method for each property.
        
        Args:
            smiles: SMILES string of the compound
            
        Returns:
            Dictionary containing all property predictions with method information
        """
        if not is_valid_smiles(smiles):
            return {'error': 'Invalid SMILES string'}
        
        print(f"Combined prediction for: {smiles}")
        
        # Get predictions from both models
        accurate_results = self.accurate.predict_all_properties(smiles)
        moleculenet_results = self.moleculenet.predict_all_properties(smiles)
        
        if accurate_results is None:
            return {'error': 'Accurate predictor failed'}
        
        # Combine results intelligently
        combined_properties = {}
        method_used = {}
        
        for prop_name, prop_config in self.property_mapping.items():
            primary_method = prop_config['primary']
            fallback_method = prop_config['fallback']
            
            # Try primary method first
            if primary_method == 'moleculenet':
                if (moleculenet_results and 
                    'properties' in moleculenet_results and 
                    prop_name in moleculenet_results['properties']):
                    
                    combined_properties[prop_name] = moleculenet_results['properties'][prop_name]
                    method_used[prop_name] = 'moleculenet'
                    value = combined_properties[prop_name]['value']
                    if isinstance(value, (int, float)):
                        print(f"✓ {prop_name}: MoleculeNet ({value:.4f})")
                    else:
                        print(f"✓ {prop_name}: MoleculeNet ({value})")
                    continue
            
            elif primary_method == 'accurate':
                if (accurate_results and 
                    'properties' in accurate_results and 
                    prop_name in accurate_results['properties']):
                    
                    combined_properties[prop_name] = accurate_results['properties'][prop_name]
                    method_used[prop_name] = 'accurate'
                    value = combined_properties[prop_name]['value']
                    if isinstance(value, (int, float)):
                        print(f"✓ {prop_name}: RDKit ({value:.4f})")
                    else:
                        print(f"✓ {prop_name}: RDKit ({value})")
                    continue
            
            # Try fallback method if primary failed
            if fallback_method == 'accurate':
                if (accurate_results and 
                    'properties' in accurate_results and 
                    prop_name in accurate_results['properties']):
                    
                    combined_properties[prop_name] = accurate_results['properties'][prop_name]
                    method_used[prop_name] = 'accurate (fallback)'
                    value = combined_properties[prop_name]['value']
                    if isinstance(value, (int, float)):
                        print(f"⚠️  {prop_name}: RDKit fallback ({value:.4f})")
                    else:
                        print(f"⚠️  {prop_name}: RDKit fallback ({value})")
                    continue
            
            print(f"❌ {prop_name}: No prediction available")
        
        return {
            'smiles': smiles,
            'properties': combined_properties,
            'method_used': method_used,
            'summary': {
                'total_properties': len(combined_properties),
                'moleculenet_properties': len([m for m in method_used.values() if 'moleculenet' in m]),
                'rdkit_properties': len([m for m in method_used.values() if 'accurate' in m]),
                'available_properties': list(combined_properties.keys())
            }
        }
    
    def get_property_comparison(self, smiles: str) -> Dict:
        """
        Get a comparison of predictions from both models for overlapping properties.
        
        Args:
            smiles: SMILES string of the compound
            
        Returns:
            Dictionary containing comparison of predictions
        """
        if not is_valid_smiles(smiles):
            return {'error': 'Invalid SMILES string'}
        
        # Get predictions from both models
        accurate_results = self.accurate.predict_all_properties(smiles)
        moleculenet_results = self.moleculenet.predict_all_properties(smiles)
        
        if accurate_results is None:
            return {'error': 'Accurate predictor failed'}
        
        comparison = {
            'smiles': smiles,
            'overlapping_properties': {},
            'moleculenet_only': {},
            'rdkit_only': {}
        }
        
        # Find overlapping properties
        accurate_props = set(accurate_results.get('properties', {}).keys())
        moleculenet_props = set(moleculenet_results.get('properties', {}).keys())
        
        overlapping = accurate_props.intersection(moleculenet_props)
        
        for prop in overlapping:
            acc_val = accurate_results['properties'][prop]['value']
            mol_val = moleculenet_results['properties'][prop]['value']
            
            # Calculate difference only for numeric values
            difference = None
            if isinstance(acc_val, (int, float)) and isinstance(mol_val, (int, float)):
                difference = abs(acc_val - mol_val)
            
            comparison['overlapping_properties'][prop] = {
                'rdkit_value': acc_val,
                'moleculenet_value': mol_val,
                'difference': difference,
                'rdkit_unit': accurate_results['properties'][prop]['unit'],
                'moleculenet_unit': moleculenet_results['properties'][prop]['unit']
            }
        
        # MoleculeNet only properties
        for prop in moleculenet_props - accurate_props:
            comparison['moleculenet_only'][prop] = moleculenet_results['properties'][prop]
        
        # RDKit only properties
        for prop in accurate_props - moleculenet_props:
            comparison['rdkit_only'][prop] = accurate_results['properties'][prop]
        
        return comparison
    
    def print_combined_report(self, results: Dict):
        """Print a formatted report for combined predictions."""
        if 'error' in results:
            print(f"❌ Error: {results['error']}")
            return
        
        print(f"\n{'='*80}")
        print(f"COMBINED PFAS PROPERTY PREDICTION REPORT")
        print(f"{'='*80}")
        print(f"SMILES: {results['smiles']}")
        print(f"{'='*80}")
        
        # Group properties by method
        moleculenet_props = []
        rdkit_props = []
        
        for prop_name, method in results['method_used'].items():
            if 'moleculenet' in method:
                moleculenet_props.append(prop_name)
            else:
                rdkit_props.append(prop_name)
        
        # Print MoleculeNet properties
        if moleculenet_props:
            print(f"\nMOLECULENET PREDICTIONS ({len(moleculenet_props)} properties):")
            print("-" * 50)
            for prop_name in moleculenet_props:
                prop_data = results['properties'][prop_name]
                value = prop_data['value']
                if isinstance(value, (int, float)):
                    print(f"  {prop_name}: {value:.4f} {prop_data['unit']}")
                else:
                    print(f"  {prop_name}: {value} {prop_data['unit']}")
        
        # Print RDKit properties
        if rdkit_props:
            print(f"\nRDKIT PREDICTIONS ({len(rdkit_props)} properties):")
            print("-" * 50)
            for prop_name in rdkit_props:
                prop_data = results['properties'][prop_name]
                value = prop_data['value']
                if isinstance(value, (int, float)):
                    print(f"  {prop_name}: {value:.4f} {prop_data['unit']}")
                else:
                    print(f"  {prop_name}: {value} {prop_data['unit']}")
        
        # Print summary
        print(f"\nSUMMARY:")
        print("-" * 50)
        print(f"  Total properties predicted: {results['summary']['total_properties']}")
        print(f"  MoleculeNet properties: {results['summary']['moleculenet_properties']}")
        print(f"  RDKit properties: {results['summary']['rdkit_properties']}")
        print(f"  Available properties: {', '.join(results['summary']['available_properties'])}")
        
        print(f"\n{'='*80}")


def main():
    """Demo the combined PFAS predictor."""
    print("Combined PFAS Property Predictor Demo")
    print("=" * 60)
    
    # Initialize predictor
    predictor = CombinedPFASPredictor()
    
    # Test with PFAS compounds
    pfas_compounds = {
        "PFOA": "C(=O)(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O",
        "PFOS": "C(C(C(C(C(F)(F)S(=O)(=O)O)(F)F)(F)F)(F)F)(C(C(C(F)(F)F)(F)F)(F)F)(F)F",
    }
    
    print(f"\nProperty mapping strategy:")
    print(f"  MoleculeNet: {len([p for p, c in predictor.property_mapping.items() if c['primary'] == 'moleculenet'])} properties")
    print(f"  RDKit: {len([p for p, c in predictor.property_mapping.items() if c['primary'] == 'accurate'])} properties")
    
    # Predict all properties for each compound
    for name, smiles in pfas_compounds.items():
        print(f"\n{'='*50}")
        print(f"Analyzing {name} with combined approach...")
        print(f"{'='*50}")
        
        # Get combined predictions
        results = predictor.predict_all(smiles)
        
        # Print detailed report
        predictor.print_combined_report(results)
        
        # Get comparison
        comparison = predictor.get_property_comparison(smiles)
        if 'overlapping_properties' in comparison and comparison['overlapping_properties']:
            print(f"\nPROPERTY COMPARISON:")
            print("-" * 30)
            for prop, comp_data in comparison['overlapping_properties'].items():
                print(f"  {prop}:")
                rdkit_val = comp_data['rdkit_value']
                molnet_val = comp_data['moleculenet_value']
                
                if isinstance(rdkit_val, (int, float)):
                    print(f"    RDKit: {rdkit_val:.4f} {comp_data['rdkit_unit']}")
                else:
                    print(f"    RDKit: {rdkit_val} {comp_data['rdkit_unit']}")
                    
                if isinstance(molnet_val, (int, float)):
                    print(f"    MoleculeNet: {molnet_val:.4f} {comp_data['moleculenet_unit']}")
                else:
                    print(f"    MoleculeNet: {molnet_val} {comp_data['moleculenet_unit']}")
                    
                if comp_data['difference'] is not None:
                    print(f"    Difference: {comp_data['difference']:.4f}")
                else:
                    print(f"    Difference: N/A (non-numeric values)")
    
    print(f"\nDemo completed successfully! 🎉")


if __name__ == "__main__":
    main() 
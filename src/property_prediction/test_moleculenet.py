"""
Simple MoleculeNet Test

This script tests if MoleculeNet/DeepChem can be imported and used
for basic molecular property prediction.

Author: PFAS AI Project
Date: 2024
"""

import sys
import os

def test_deepchem_import():
    """Test if DeepChem can be imported."""
    print("Testing DeepChem import...")
    
    try:
        import deepchem as dc
        print(f"✓ DeepChem imported successfully")
        print(f"  - Version: {dc.__version__}")
        return True
    except ImportError as e:
        print(f"❌ DeepChem import failed: {e}")
        print("  Install with: pip install deepchem")
        return False

def test_rdkit_import():
    """Test if RDKit can be imported."""
    print("\nTesting RDKit import...")
    
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        print(f"✓ RDKit imported successfully")
        print(f"  - Version: {Chem.rdMolDescriptors._version}")
        return True
    except ImportError as e:
        print(f"❌ RDKit import failed: {e}")
        print("  Install with: conda install -c conda-forge rdkit")
        return False

def test_moleculenet_datasets():
    """Test if MoleculeNet datasets can be loaded."""
    print("\nTesting MoleculeNet datasets...")
    
    try:
        import deepchem as dc
        
        # Test loading a small dataset
        print("  Loading Delaney (ESOL) dataset...")
        tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='ECFP', split='random')
        train_dataset, valid_dataset, test_dataset = datasets
        
        print(f"✓ Dataset loaded successfully")
        print(f"  - Training samples: {len(train_dataset)}")
        print(f"  - Validation samples: {len(valid_dataset)}")
        print(f"  - Test samples: {len(test_dataset)}")
        print(f"  - Number of tasks: {len(tasks)}")
        print(f"  - Task names: {tasks}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False

def test_simple_prediction():
    """Test a simple prediction with MoleculeNet."""
    print("\nTesting simple prediction...")
    
    try:
        import deepchem as dc
        from rdkit import Chem
        
        # Create a simple molecule (ethanol)
        smiles = "CCO"
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            print("❌ Failed to create molecule from SMILES")
            return False
            
        print(f"✓ Created molecule: {smiles}")
        
        # Calculate a simple descriptor
        mw = Chem.Descriptors.MolWt(mol)
        logp = Chem.Descriptors.MolLogP(mol)
        
        print(f"  - Molecular weight: {mw:.2f}")
        print(f"  - LogP: {logp:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Simple prediction failed: {e}")
        return False

def test_pfas_compounds():
    """Test with PFAS compounds."""
    print("\nTesting PFAS compounds...")
    
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        
        # PFAS compounds
        pfas_smiles = [
            "C(=O)(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O",  # PFOA
            "C(C(C(C(C(F)(F)S(=O)(=O)O)(F)F)(F)F)(F)F)(C(C(C(F)(F)F)(F)F)(F)F)(F)F",  # PFOS
        ]
        
        pfas_names = ["PFOA", "PFOS"]
        
        for i, (smiles, name) in enumerate(zip(pfas_smiles, pfas_names)):
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is None:
                print(f"❌ Failed to create {name} molecule")
                continue
                
            mw = Chem.Descriptors.MolWt(mol)
            logp = Chem.Descriptors.MolLogP(mol)
            tpsa = Chem.Descriptors.TPSA(mol)
            
            print(f"✓ {name}:")
            print(f"  - SMILES: {smiles}")
            print(f"  - Molecular weight: {mw:.2f}")
            print(f"  - LogP: {logp:.2f}")
            print(f"  - TPSA: {tpsa:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ PFAS testing failed: {e}")
        return False

def main():
    """Run all tests."""
    print("MoleculeNet/DeepChem Test Suite")
    print("=" * 40)
    
    tests = [
        ("DeepChem Import", test_deepchem_import),
        ("RDKit Import", test_rdkit_import),
        ("MoleculeNet Datasets", test_moleculenet_datasets),
        ("Simple Prediction", test_simple_prediction),
        ("PFAS Compounds", test_pfas_compounds)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*15} {test_name} {'='*15}")
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*40}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! MoleculeNet is working correctly.")
    else:
        print("⚠️  Some tests failed. Please install missing dependencies.")
    
    return passed == total

if __name__ == "__main__":
    main() 
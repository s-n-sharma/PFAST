from pfas_prediction_models.combined_pfas import CombinedPFASPredictor

def test_combined_pfas_predictor():
    """Test Combined PFAS predictor functionality."""
    print("Testing Combined PFAS Predictor...")
    
    # Initialize predictor
    predictor = CombinedPFASPredictor()
    
    # Test SMILES
    smiles = "C(=O)(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O"  # PFOA
    
    # Test combined prediction
    result = predictor.predict_all(smiles)
    
    # Basic assertions
    assert result is not None, "Result should not be None"
    assert 'smiles' in result, "Result should contain SMILES"
    assert 'properties' in result, "Result should contain properties"
    assert 'method_used' in result, "Result should contain method information"
    assert 'summary' in result, "Result should contain summary"
    
    # Check properties
    if result['properties']:
        print(f"✓ Successfully predicted {len(result['properties'])} properties")
        
        # Count methods used
        moleculenet_count = len([m for m in result['method_used'].values() if 'moleculenet' in m])
        rdkit_count = len([m for m in result['method_used'].values() if 'accurate' in m])
        
        print(f"  - MoleculeNet predictions: {moleculenet_count}")
        print(f"  - RDKit predictions: {rdkit_count}")
        
        # Check a few key properties
        for prop_name, prop_data in result['properties'].items():
            assert 'value' in prop_data, f"Property {prop_name} should have a value"
            assert 'unit' in prop_data, f"Property {prop_name} should have a unit"
            method = result['method_used'].get(prop_name, 'unknown')
            value = prop_data['value']
            if isinstance(value, (int, float)):
                print(f"  - {prop_name}: {value:.4f} {prop_data['unit']} ({method})")
            else:
                print(f"  - {prop_name}: {value} {prop_data['unit']} ({method})")
    else:
        print("⚠️  No properties predicted")
    
    # Check summary
    summary = result['summary']
    assert summary['total_properties'] >= 0, "Should have property count"
    print(f"✓ Summary: {summary['total_properties']} total properties")
    
    # Test property comparison
    comparison = predictor.get_property_comparison(smiles)
    assert comparison is not None, "Comparison should not be None"
    assert 'overlapping_properties' in comparison, "Should have overlapping properties"
    assert 'moleculenet_only' in comparison, "Should have MoleculeNet-only properties"
    assert 'rdkit_only' in comparison, "Should have RDKit-only properties"
    
    print(f"✓ Comparison: {len(comparison['overlapping_properties'])} overlapping properties")
    
    print("Combined PFAS Predictor test passed.")

if __name__ == "__main__":
    test_combined_pfas_predictor() 
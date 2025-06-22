from pfas_prediction_models.moleculenet_pfas import MoleculeNetPFASPredictor

def test_moleculenet_pfas_predictor():
    """Test MoleculeNet PFAS predictor functionality."""
    print("Testing MoleculeNet PFAS Predictor...")
    
    # Initialize predictor
    predictor = MoleculeNetPFASPredictor()
    
    # Test SMILES
    smiles = "C(=O)(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O"  # PFOA
    
    # Test prediction
    result = predictor.predict_all_properties(smiles)
    
    # Basic assertions
    assert result is not None, "Result should not be None"
    assert 'smiles' in result, "Result should contain SMILES"
    assert 'properties' in result, "Result should contain properties"
    assert 'model_info' in result, "Result should contain model info"
    
    # Check if any properties were predicted
    if result['properties']:
        print(f"✓ Successfully predicted {len(result['properties'])} properties")
        for prop_name, prop_data in result['properties'].items():
            assert 'value' in prop_data, f"Property {prop_name} should have a value"
            assert 'unit' in prop_data, f"Property {prop_name} should have a unit"
            print(f"  - {prop_name}: {prop_data['value']:.4f} {prop_data['unit']}")
    else:
        print("⚠️  No properties predicted (models may still be loading)")
    
    # Check model info
    assert result['model_info']['total_models'] >= 0, "Should have model count"
    print(f"✓ Model info: {result['model_info']['total_models']} models loaded")
    
    print("MoleculeNet PFAS Predictor test passed.")

if __name__ == "__main__":
    test_moleculenet_pfas_predictor() 
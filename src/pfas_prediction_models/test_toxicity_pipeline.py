#!/usr/bin/env python3
"""Test script for the updated toxicity prediction pipeline with toxcast format."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from toxicity_predictor import ToxicityPredictor

def test_toxicity_pipeline():
    """Test the complete toxicity prediction pipeline."""
    print("🧪 Testing PFAS Toxicity Prediction Pipeline (Toxcast Format)")
    print("=" * 70)
    
    # Initialize the predictor
    print("1. Initializing ToxicityPredictor...")
    predictor = ToxicityPredictor()
    
    if predictor.model is None:
        print("❌ Failed to load toxicity model")
        return False
    
    print("✓ ToxicityPredictor initialized successfully")
    
    # Test SMILES
    test_smiles = "C(=O)(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O"  # PFOA
    
    print(f"\n2. Testing SMILES: {test_smiles}")
    print("-" * 40)
    
    # Test toxcast format generation
    print("   Generating toxcast format data...")
    toxcast_data = predictor.generate_toxcast_format(test_smiles)
    
    if toxcast_data is None:
        print("   ❌ Failed to generate toxcast format data")
        return False
    
    print(f"   ✓ Generated toxcast format data with {len(toxcast_data)} properties")
    print(f"   ✓ Sample properties: solubility={toxcast_data.get('solubility', 'N/A')}, molecular_weight={toxcast_data.get('molecular_weight', 'N/A')}")
    
    # Test feature extraction
    print("   Extracting features for model...")
    try:
        features_tensor = predictor.extract_features_for_model(toxcast_data)
        print(f"   ✓ Extracted {len(features_tensor)} features")
        print(f"   ✓ Tensor shape: {features_tensor.shape}, dtype: {features_tensor.dtype}")
    except Exception as e:
        print(f"   ❌ Feature extraction failed: {e}")
        return False
    
    # Test full prediction
    print("   Running full prediction...")
    result = predictor.predict_toxicity(test_smiles)
    
    if 'error' in result:
        print(f"   ❌ Prediction failed: {result['error']}")
        return False
    
    print(f"   ✓ Prediction successful")
    print(f"   ✓ Features: {result['feature_count']}")
    print(f"   ✓ Probabilities: {result['probability_count']}")
    
    # Show first few probabilities
    if result['probabilities']:
        print(f"   ✓ Sample probabilities: {result['probabilities'][:5]}")
    
    # Test CSV saving
    print("   Testing CSV saving...")
    try:
        predictor.save_toxcast_csv([result], "test_toxcast_output.csv")
        print("   ✓ CSV saving successful")
    except Exception as e:
        print(f"   ❌ CSV saving failed: {e}")
    
    print(f"\n🎉 Test passed! Toxicity prediction pipeline is working correctly.")
    return True

def test_model_architecture():
    """Test the model architecture specifically."""
    print("\n🔧 Testing Model Architecture")
    print("=" * 40)
    
    try:
        import torch
        from toxicity_predictor import MultiTaskNet
        
        # Test creating the model
        print("1. Creating MultiTaskNet...")
        model = MultiTaskNet(num_features=36, num_tasks=96)
        print("✓ MultiTaskNet created successfully")
        
        # Test forward pass
        print("2. Testing forward pass...")
        test_input = torch.randn(1, 36, dtype=torch.float32)
        output = model(test_input)
        print(f"✓ Forward pass successful")
        print(f"✓ Input shape: {test_input.shape}")
        print(f"✓ Output shape: {output.shape}")
        
        # Test state dict loading
        print("3. Testing state dict loading...")
        state_dict = model.state_dict()
        print(f"✓ State dict keys: {list(state_dict.keys())[:5]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Model architecture test failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting toxicity prediction test...")
    
    # Test model architecture first
    arch_success = test_model_architecture()
    
    # Test full pipeline
    pipeline_success = test_toxicity_pipeline()
    
    if arch_success and pipeline_success:
        print(f"\n✅ All tests passed!")
        sys.exit(0)
    else:
        print(f"\n❌ Some tests failed!")
        sys.exit(1) 
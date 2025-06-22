"""
Test package for PFAS Property Prediction Models
"""

from .test_accurate import test_accurate_pfas_predictor
from .test_moleculenet import test_moleculenet_pfas_predictor
from .test_combined import test_combined_pfas_predictor

def run_all_tests():
    """Run all tests in the package."""
    print("Running PFAS Property Prediction Models tests...")
    
    try:
        test_accurate_pfas_predictor()
        print("✓ Accurate PFAS Predictor test passed")
    except Exception as e:
        print(f"✗ Accurate PFAS Predictor test failed: {e}")
    
    try:
        test_moleculenet_pfas_predictor()
        print("✓ MoleculeNet PFAS Predictor test passed")
    except Exception as e:
        print(f"✗ MoleculeNet PFAS Predictor test failed: {e}")
    
    try:
        test_combined_pfas_predictor()
        print("✓ Combined PFAS Predictor test passed")
    except Exception as e:
        print(f"✗ Combined PFAS Predictor test failed: {e}")
    
    print("All tests completed!")

if __name__ == "__main__":
    run_all_tests() 
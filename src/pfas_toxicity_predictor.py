"""
PFAS Toxicity Prediction System

End-to-end system that takes a SMILES string, generates property data,
converts to feature vector, and predicts toxicity using a trained PyTorch model.

Author: PFAS AI Project
Date: 2024
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Add the current directory to the path to import the predictors
sys.path.append(os.path.dirname(__file__))

from pfas_prediction_models.combined_pfas import CombinedPFASPredictor


class ToxicityPredictor:
    """
    Complete toxicity prediction system for PFAS compounds.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the toxicity predictor.
        
        Args:
            model_path: Path to the PyTorch model file
        """
        self.property_predictor = CombinedPFASPredictor()
        
        # Default model path
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__), 
                'bestmodel.pth'
            )
        
        self.model_path = model_path
        self.model = None
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the PyTorch toxicity prediction model."""
        try:
            print(f"Loading toxicity model from: {self.model_path}")
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Load the model
            self.model = torch.load(self.model_path, map_location=torch.device('cpu'))
            self.model.eval()
            
            print("✓ Toxicity model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
    
    def extract_numeric_features(self, properties: Dict) -> List[float]:
        """
        Extract numeric features from property predictions.
        Filters out non-numeric data and ensures 35 features.
        
        Args:
            properties: Dictionary of property predictions
            
        Returns:
            List of 35 numeric features
        """
        numeric_features = []
        
        # Define the order of properties to extract (35 features)
        property_order = [
            'molecular_weight', 'logp', 'logd', 'tpsa', 'hbd', 'hba',
            'rotatable_bonds', 'aromatic_rings', 'saturated_rings', 'heteroatoms',
            'fluorine_count', 'fluorine_percentage', 'carbon_count', 'oxygen_count',
            'sulfur_count', 'nitrogen_count', 'hydrogen_count', 'amide_bonds',
            'ester_bonds', 'ether_bonds', 'sulfonamides', 'sulfones',
            'fraction_csp3', 'spiro_atoms', 'bridgehead_atoms', 'drug_likeness',
            'bioaccumulation_potential', 'environmental_persistence', 'toxicity_risk',
            'solubility', 'lipophilicity', 'bbbp', 'hiv'
        ]
        
        # Extract numeric values in order
        for prop_name in property_order:
            if prop_name in properties:
                value = properties[prop_name]['value']
                
                # Convert to float, handle non-numeric values
                if isinstance(value, (int, float)):
                    numeric_features.append(float(value))
                elif isinstance(value, str):
                    # Try to extract numeric part from string (e.g., "C8HF15O2" -> 8+1+15+2 = 26)
                    try:
                        # Simple atom counting for molecular formula
                        if prop_name == 'formula':
                            # Count atoms in formula
                            count = 0
                            for char in value:
                                if char.isdigit():
                                    count += int(char)
                                elif char.isupper():
                                    count += 1
                            numeric_features.append(float(count))
                        else:
                            numeric_features.append(0.0)  # Default for non-numeric strings
                    except:
                        numeric_features.append(0.0)
                else:
                    numeric_features.append(0.0)
            else:
                # Property not available, use default value
                numeric_features.append(0.0)
        
        # Ensure we have exactly 35 features
        while len(numeric_features) < 35:
            print(f"Adding default value for feature {len(numeric_features) + 1}")
            numeric_features.append(0.0)
        
        if len(numeric_features) > 35:
            numeric_features = numeric_features[:35]
        
        return numeric_features
    
    def predict_toxicity(self, smiles: str) -> Dict:
        """
        Complete toxicity prediction pipeline.
        
        Args:
            smiles: SMILES string of the PFAS compound
            
        Returns:
            Dictionary containing prediction results
        """
        print(f"Predicting toxicity for: {smiles}")
        print("=" * 50)
        
        # Step 1: Generate property predictions
        print("Step 1: Generating molecular properties...")
        property_result = self.property_predictor.predict_all(smiles)
        
        if not property_result or 'properties' not in property_result:
            return {
                'error': 'Failed to generate property predictions',
                'smiles': smiles
            }
        
        properties = property_result['properties']
        print(f"✓ Generated {len(properties)} properties")
        
        # Step 2: Extract numeric features
        print("Step 2: Extracting numeric features...")
        features = self.extract_numeric_features(properties)
        print(f"✓ Extracted {len(features)} numeric features")
        
        # Step 3: Predict toxicity using proptotoxic2.py
        print("Step 3: Predicting toxicity...")
        if self.model is None:
            return {
                'error': 'Toxicity model not loaded',
                'smiles': smiles,
                'features': features
            }
        
        try:
            # Convert features to tensor
            feature_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            
            # Make prediction using the model
            with torch.no_grad():
                prediction = self.model(feature_tensor)
                
                # Get probabilities (should be 96 outputs)
                if prediction.shape[1] == 96:
                    probabilities = torch.softmax(prediction, dim=1)
                    prob_list = probabilities[0].tolist()
                else:
                    # If not 96 outputs, use sigmoid for binary or softmax for multi-class
                    if prediction.shape[1] == 1:
                        prob_list = [torch.sigmoid(prediction).item()]
                    else:
                        probabilities = torch.softmax(prediction, dim=1)
                        prob_list = probabilities[0].tolist()
            
            print(f"✓ Toxicity prediction completed")
            
            return {
                'smiles': smiles,
                'probabilities': prob_list,
                'features': features,
                'properties': properties,
                'feature_count': len(features),
                'probability_count': len(prob_list)
            }
            
        except Exception as e:
            return {
                'error': f'Toxicity prediction failed: {str(e)}',
                'smiles': smiles,
                'features': features
            }
    
    def print_prediction_report(self, result: Dict):
        """Print a formatted prediction report."""
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            return
        
        print(f"\n{'='*60}")
        print(f"PFAS TOXICITY PREDICTION REPORT")
        print(f"{'='*60}")
        print(f"SMILES: {result['smiles']}")
        print(f"Feature Count: {result['feature_count']}")
        print(f"Probability Count: {result['probability_count']}")
        
        print(f"\nFeature Vector (35 values):")
        features = result['features']
        for i, feature in enumerate(features):
            print(f"  Feature {i+1:2d}: {feature:.4f}")
        
        print(f"\nToxicity Probabilities ({len(result['probabilities'])} values):")
        probabilities = result['probabilities']
        for i, prob in enumerate(probabilities):
            print(f"  Prob {i+1:2d}: {prob:.6f}")
        
        print(f"\n{'='*60}")


def main():
    """Demo the complete toxicity prediction system."""
    print("PFAS Toxicity Prediction System")
    print("=" * 50)
    
    # Initialize predictor
    predictor = ToxicityPredictor()
    
    # Test SMILES
    test_smiles = [
        "C(=O)(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O",  # PFOA
        "C(C(C(C(C(F)(F)S(=O)(=O)O)(F)F)(F)F)(F)F)(C(C(C(F)(F)F)(F)F)(F)F)(F)F",  # PFOS
        "C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(C(=O)O)C(=O)O",  # GenX
    ]
    
    for smiles in test_smiles:
        print(f"\n{'='*50}")
        result = predictor.predict_toxicity(smiles)
        predictor.print_prediction_report(result)
    
    print(f"\n🎉 Toxicity prediction demo completed!")


if __name__ == "__main__":
    main() 
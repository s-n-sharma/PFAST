"""
PFAS Toxicity Predictor

Generates property predictions in toxcast_predictions.csv format and feeds them
to the proptotoxic2.py MultiTaskNet model for toxicity prediction.

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

from combined_pfas import CombinedPFASPredictor


class MultiTaskNet(nn.Module):
    """Multi-task neural network with Batch Normalization for improved training."""
    def __init__(self, num_features, num_tasks, dropout_rate=0.5):
        super(MultiTaskNet, self).__init__()
        self.network = nn.Sequential(
            # Block 1
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Block 2
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            
            # Block 3
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            # Output Layer
            nn.Linear(256, num_tasks)
        )

    def forward(self, x):
        return self.network(x)


class ToxicityPredictor:
    """
    Toxicity prediction system that generates features in toxcast_predictions.csv format
    and feeds them to the proptotoxic2.py model.
    """
    
    def __init__(self):
        """
        Initialize the toxicity predictor.
        """
        self.property_predictor = CombinedPFASPredictor()
        
        # Load the model
        try:
            print("Loading toxicity model...")
            
            # Load the state dictionary
            state_dict = torch.load('bestmodel.pth', map_location=torch.device('cpu'))
            
            # Determine input and output sizes from the state dict
            # The first layer weight shape will tell us input size
            first_layer_weight = state_dict['network.0.weight']
            last_layer_weight = state_dict['network.11.weight']
            
            num_features = first_layer_weight.shape[1]  # Input features
            num_tasks = last_layer_weight.shape[0]      # Output tasks
            
            print(f"Model architecture: {num_features} features -> {num_tasks} tasks")
            
            # Create the model with correct architecture
            self.model = MultiTaskNet(num_features, num_tasks)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            print("✓ Toxicity model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
    
    def generate_toxcast_format(self, smiles: str) -> Dict:
        """
        Generate property predictions in the exact same format as toxcast_predictions.csv.
        
        Args:
            smiles: SMILES string of the PFAS compound
            
        Returns:
            Dictionary with all properties in toxcast format
        """
        # Get property predictions from combined predictor
        property_result = self.property_predictor.predict_all(smiles)
        
        if not property_result or 'properties' not in property_result:
            return None
        
        properties = property_result['properties']
        
        # Create toxcast format dictionary
        toxcast_data = {
            'smiles': smiles,
            # MoleculeNet predictions (first 4 columns)
            'solubility': properties.get('solubility', {}).get('value', 0.0),
            'lipophilicity': properties.get('lipophilicity', {}).get('value', 0.0),
            'bbbp': properties.get('bbbp', {}).get('value', 0.0),
            'hiv': properties.get('hiv', {}).get('value', 0.0),
            # Structural properties (rest of columns)
            'molecular_weight': properties.get('molecular_weight', {}).get('value', 0.0),
            'formula': properties.get('formula', {}).get('value', ''),
            'logp': properties.get('logp', {}).get('value', 0.0),
            'logd': properties.get('logd', {}).get('value', 0.0),
            'tpsa': properties.get('tpsa', {}).get('value', 0.0),
            'hbd': properties.get('hbd', {}).get('value', 0.0),
            'hba': properties.get('hba', {}).get('value', 0.0),
            'rotatable_bonds': properties.get('rotatable_bonds', {}).get('value', 0.0),
            'aromatic_rings': properties.get('aromatic_rings', {}).get('value', 0.0),
            'saturated_rings': properties.get('saturated_rings', {}).get('value', 0.0),
            'heteroatoms': properties.get('heteroatoms', {}).get('value', 0.0),
            'fluorine_count': properties.get('fluorine_count', {}).get('value', 0.0),
            'fluorine_percentage': properties.get('fluorine_percentage', {}).get('value', 0.0),
            'carbon_count': properties.get('carbon_count', {}).get('value', 0.0),
            'oxygen_count': properties.get('oxygen_count', {}).get('value', 0.0),
            'sulfur_count': properties.get('sulfur_count', {}).get('value', 0.0),
            'nitrogen_count': properties.get('nitrogen_count', {}).get('value', 0.0),
            'hydrogen_count': properties.get('hydrogen_count', {}).get('value', 0.0),
            'amide_bonds': properties.get('amide_bonds', {}).get('value', 0.0),
            'ester_bonds': properties.get('ester_bonds', {}).get('value', 0.0),
            'ether_bonds': properties.get('ether_bonds', {}).get('value', 0.0),
            'sulfonamides': properties.get('sulfonamides', {}).get('value', 0.0),
            'sulfones': properties.get('sulfones', {}).get('value', 0.0),
            'fraction_csp3': properties.get('fraction_csp3', {}).get('value', 0.0),
            'spiro_atoms': properties.get('spiro_atoms', {}).get('value', 0.0),
            'bridgehead_atoms': properties.get('bridgehead_atoms', {}).get('value', 0.0),
            'drug_likeness': properties.get('drug_likeness', {}).get('value', 0.0),
            'bioaccumulation_potential': properties.get('bioaccumulation_potential', {}).get('value', 0.0),
            'environmental_persistence': properties.get('environmental_persistence', {}).get('value', 0.0),
            'toxicity_risk': properties.get('toxicity_risk', {}).get('value', 0.0)
        }
        
        return toxcast_data
    
    def extract_features_for_model(self, toxcast_data: Dict) -> torch.Tensor:
        """
        Extract features from toxcast format data for the model.
        This excludes the 'smiles' and 'formula' columns and converts to tensor.
        
        Args:
            toxcast_data: Dictionary in toxcast format
            
        Returns:
            torch.Tensor of features with dtype=torch.float32
        """
        # Define the feature columns in order (excluding smiles and formula)
        feature_columns = [
            'solubility', 'lipophilicity', 'bbbp', 'hiv', 'molecular_weight',
            'logp', 'logd', 'tpsa', 'hbd', 'hba', 'rotatable_bonds', 'aromatic_rings',
            'saturated_rings', 'heteroatoms', 'fluorine_count', 'fluorine_percentage',
            'carbon_count', 'oxygen_count', 'sulfur_count', 'nitrogen_count',
            'hydrogen_count', 'amide_bonds', 'ester_bonds', 'ether_bonds',
            'sulfonamides', 'sulfones', 'fraction_csp3', 'spiro_atoms',
            'bridgehead_atoms', 'drug_likeness', 'bioaccumulation_potential',
            'environmental_persistence', 'toxicity_risk'
        ]
        
        # Extract features in order
        features = []
        for col in feature_columns:
            value = toxcast_data.get(col, 0.0)
            
            # Convert to float, handle non-numeric values
            if isinstance(value, (int, float)):
                features.append(float(value))
            else:
                features.append(0.0)
        
        # Convert to tensor with proper dtype like in proptotoxic2.py
        features_tensor = torch.tensor(features, dtype=torch.float32)
        
        return features_tensor
    
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
        
        # Step 1: Generate toxcast format data
        print("Step 1: Generating toxcast format properties...")
        toxcast_data = self.generate_toxcast_format(smiles)
        
        if toxcast_data is None:
            return {
                'error': 'Failed to generate toxcast format data',
                'smiles': smiles
            }
        
        print(f"✓ Generated toxcast format data with {len(toxcast_data)} properties")
        
        # Step 2: Extract features for model
        print("Step 2: Extracting features for model...")
        features_tensor = self.extract_features_for_model(toxcast_data)
        print(f"✓ Extracted {len(features_tensor)} features for model")
        
        # Step 3: Predict toxicity
        print("Step 3: Predicting toxicity...")
        if self.model is None:
            return {
                'error': 'Toxicity model not loaded',
                'smiles': smiles,
                'toxcast_data': toxcast_data,
                'features': features_tensor.tolist()
            }
        
        try:
            # Add batch dimension for prediction
            feature_tensor_batch = features_tensor.unsqueeze(0)
            
            # Make prediction
            with torch.no_grad():
                prediction = self.model(feature_tensor_batch)
                
                # Apply sigmoid to get probabilities (since it's a binary classification task)
                probabilities = torch.sigmoid(prediction)
                prob_list = probabilities[0].tolist()
            
            print(f"✓ Toxicity prediction completed")
            
            return {
                'smiles': smiles,
                'toxcast_data': toxcast_data,
                'features': features_tensor.tolist(),
                'probabilities': prob_list,
                'feature_count': len(features_tensor),
                'probability_count': len(prob_list)
            }
            
        except Exception as e:
            return {
                'error': f'Toxicity prediction failed: {str(e)}',
                'smiles': smiles,
                'toxcast_data': toxcast_data,
                'features': features_tensor.tolist()
            }
    
    def save_toxcast_csv(self, results: List[Dict], filename: str = None):
        """
        Save results to CSV in toxcast_predictions format.
        
        Args:
            results: List of prediction results
            filename: Output filename (optional)
        """
        if not results:
            print("No results to save")
            return
        
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"toxcast_predictions_{timestamp}.csv"
        
        # Extract toxcast data from results
        toxcast_data_list = []
        for result in results:
            if 'toxcast_data' in result:
                toxcast_data_list.append(result['toxcast_data'])
        
        if toxcast_data_list:
            df = pd.DataFrame(toxcast_data_list)
            df.to_csv(filename, index=False)
            print(f"✓ Saved {len(df)} predictions to {filename}")
        else:
            print("No toxcast data to save")
    
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
        
        # Show toxcast format data
        if 'toxcast_data' in result:
            print(f"\nToxcast Format Data:")
            toxcast_data = result['toxcast_data']
            for key, value in toxcast_data.items():
                if key != 'smiles' and key != 'formula':
                    print(f"  {key}: {value}")
        
        print(f"\nToxicity Probabilities ({len(result['probabilities'])} values):")
        probabilities = result['probabilities']
        for i, prob in enumerate(probabilities):
            print(f"  Prob {i+1:2d}: {prob:.6f}")
        
        print(f"\n{'='*60}")


def main():
    """Demo the toxicity prediction system."""
    print("PFAS Toxicity Prediction System")
    print("=" * 50)
    
    # Initialize predictor
    predictor = ToxicityPredictor()
    
    # Test SMILES
    test_smiles = [
        "C(=O)(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O",  # PFOA
        "C(C(C(C(C(F)(F)S(=O)(=O)O)(F)F)(F)F)(F)F)(C(C(C(F)(F)F)(F)F)(F)F)(F)F",  # PFOS
    ]
    
    results = []
    for smiles in test_smiles:
        print(f"\n{'='*50}")
        result = predictor.predict_toxicity(smiles)
        predictor.print_prediction_report(result)
        results.append(result)
    
    # Save results to CSV
    predictor.save_toxcast_csv(results)
    
    print(f"\n🎉 Toxicity prediction demo completed!")


if __name__ == "__main__":
    main() 
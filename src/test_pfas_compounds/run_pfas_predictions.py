"""
PFAS Property Prediction Script

Reads PFAS SMILES from input file and runs the combined predictor
to generate property predictions for all compounds.

Author: PFAS AI Project
Date: 2024
"""

import sys
import os
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add the parent directory to the path to import the predictors
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pfas_prediction_models.combined_pfas import CombinedPFASPredictor


def read_pfas_smiles(input_file):
    """
    Read PFAS SMILES from input file.
    
    Args:
        input_file: Path to the input file containing SMILES
        
    Returns:
        Dictionary mapping compound names to SMILES strings
    """
    compounds = {}
    
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Parse compound name and SMILES
            if ':' in line:
                parts = line.split(':', 1)
                compound_name = parts[0].strip()
                smiles = parts[1].strip()
                compounds[compound_name] = smiles
    
    return compounds


def run_predictions(compounds):
    """
    Run predictions for all compounds using the combined predictor.
    
    Args:
        compounds: Dictionary mapping compound names to SMILES
        
    Returns:
        List of dictionaries containing prediction results
    """
    print("Initializing Combined PFAS Predictor...")
    predictor = CombinedPFASPredictor()
    
    results = []
    
    print(f"\nRunning predictions for {len(compounds)} compounds...")
    print("=" * 60)
    
    for i, (compound_name, smiles) in enumerate(compounds.items(), 1):
        print(f"\n[{i}/{len(compounds)}] Predicting properties for {compound_name}")
        print(f"SMILES: {smiles}")
        
        try:
            # Run prediction
            result = predictor.predict_all(smiles)
            
            if result and 'properties' in result:
                # Extract properties for CSV
                row_data = {
                    'compound_name': compound_name,
                    'smiles': smiles
                }
                
                # Add individual properties
                for prop_name, prop_data in result['properties'].items():
                    value = prop_data['value']
                    unit = prop_data['unit']
                    
                    # Store value and unit
                    row_data[f'{prop_name}_value'] = value
                    row_data[f'{prop_name}_unit'] = unit
                
                results.append(row_data)
                print(f"✓ Successfully predicted {len(result['properties'])} properties")
                
            else:
                print(f"❌ Failed to predict properties for {compound_name}")
                # Add error row
                results.append({
                    'compound_name': compound_name,
                    'smiles': smiles,
                    'error': 'Prediction failed'
                })
                
        except Exception as e:
            print(f"❌ Error predicting {compound_name}: {e}")
            # Add error row
            results.append({
                'compound_name': compound_name,
                'smiles': smiles,
                'error': str(e)
            })
    
    return results


def save_to_csv(results, output_file):
    """
    Save prediction results to CSV file.
    
    Args:
        results: List of dictionaries containing prediction results
        output_file: Path to the output CSV file
    """
    if not results:
        print("No results to save!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    print(f"\n✓ Results saved to: {output_file}")
    print(f"  - Total compounds: {len(results)}")
    print(f"  - Total columns: {len(df.columns)}")
    
    # Show basic summary
    successful_predictions = len([r for r in results if 'error' not in r])
    print(f"  - Successful predictions: {successful_predictions}")
    if len(results) - successful_predictions > 0:
        print(f"  - Failed predictions: {len(results) - successful_predictions}")


def main():
    """Main function to run PFAS property predictions."""
    print("PFAS Property Prediction Pipeline")
    print("=" * 50)
    
    # File paths
    input_file = os.path.join(os.path.dirname(__file__), 'pfas_smiles_input.txt')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(os.path.dirname(__file__), f'pfas_predictions_{timestamp}.csv')
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return
    
    # Read compounds
    print(f"Reading compounds from: {input_file}")
    compounds = read_pfas_smiles(input_file)
    
    if not compounds:
        print("❌ No compounds found in input file!")
        return
    
    print(f"✓ Found {len(compounds)} compounds:")
    for name, smiles in compounds.items():
        print(f"  - {name}: {smiles}")
    
    # Run predictions
    results = run_predictions(compounds)
    
    # Save results
    save_to_csv(results, output_file)
    
    print(f"\n🎉 Prediction pipeline completed successfully!")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main() 
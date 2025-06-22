"""
ToxCast PFAS Property Prediction Script

Reads SMILES from toxcast_data.csv and runs the combined predictor
to generate property predictions for all compounds.

Author: PFAS AI Project
Date: 2024
"""

import sys
import os
import pandas as pd
from datetime import datetime
import warnings
from typing import List
warnings.filterwarnings('ignore')

# Add the parent directory to the path to import the predictors
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pfas_prediction_models.combined_pfas import CombinedPFASPredictor
from pfas_prediction_models.accurate_pfas import AccuratePFASPredictor


class DataCreator:
        

    def read_toxcast_smiles(input_file, max_compounds=None):
        """
        Read SMILES from toxcast_data.csv file.
        
        Args:
            input_file: Path to the toxcast_data.csv file
            max_compounds: Maximum number of compounds to process (for testing)
            
        Returns:
            List of SMILES strings
        """
        print(f"Reading SMILES from: {input_file}")
        
        try:
            # Read the CSV file
            df = pd.read_csv(input_file)
            
            # Get SMILES from first column
            smiles_column = df.columns[0]
            print(f"Using column: {smiles_column}")
            
            # Extract SMILES
            smiles_list = df[smiles_column].dropna().tolist()
            
            if max_compounds:
                smiles_list = smiles_list[:max_compounds]
                print(f"Limited to first {max_compounds} compounds for testing")
            
            print(f"✓ Found {len(smiles_list)} SMILES strings")
            
            return smiles_list
            
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return []


    def get_feature_vector(smiles: str) -> List[float]:
        """
        Get feature vector for a single SMILES string in the same format as features.csv.
        
        Args:
            smiles: SMILES string
            
        Returns:
            List of 36 feature values
        """
    
        x = DataCreator.run_predictions([smiles])
        
        # x is a list, so get the first (and only) result
        if not x or len(x) == 0:
            return [0.0] * 36
        
        result = x[0]  # Get the first result from the list
        
        feature_vectors = [None] * 36
        for i, prop_name in enumerate(result.keys()):
            if prop_name != 'smiles':  # Skip the smiles column
                feature_vectors[i] = result[prop_name]
        
        
        data_frame_x = pd.DataFrame([feature_vectors])  # Wrap in list for DataFrame
        for col in data_frame_x.columns:
            data_frame_x[col] = pd.to_numeric(data_frame_x[col], errors='coerce')
        data_frame_x.fillna(0, inplace=True)
    
        data_list = data_frame_x.values.tolist()[0]  # Get the first (and only) row
        
        return data_list


    def run_predictions(smiles_list):
        """
        Run predictions for all SMILES using the combined predictor.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            List of dictionaries containing prediction results
        """
        print("Initializing Combined PFAS Predictor...")
        predictor = AccuratePFASPredictor()
        
        results = []
        
        print(f"\nRunning predictions for {len(smiles_list)} compounds...")
        print("=" * 60)
        
        for i, smiles in enumerate(smiles_list, 1):
            print(f"\n[{i}/{len(smiles_list)}] Predicting properties for compound {i}")
            print(f"SMILES: {smiles}")
            
            try:
                # Run prediction
                result = predictor.predict_all(smiles)
                
                if result and 'properties' in result:
                    # Extract properties for CSV
                    row_data = {
                        'smiles': smiles
                    }
                    
                    # Add individual properties (values only, no units)
                    for prop_name, prop_data in result['properties'].items():
                        value = prop_data['value']
                        row_data[prop_name] = value
                    
                    results.append(row_data)
                    print(f"✓ Successfully predicted {len(result['properties'])} properties")
                    
                else:
                    print(f"❌ Failed to predict properties for compound {i}")
                    # Add error row
                    results.append({
                        'smiles': smiles,
                        'error': 'Prediction failed'
                    })
                    
            except Exception as e:
                print(f"❌ Error predicting compound {i}: {e}")
                # Add error row
                results.append({
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
    """Main function to run ToxCast PFAS property predictions."""
    print("ToxCast PFAS Property Prediction Pipeline")
    print("=" * 50)
    
    # File paths
    input_file = os.path.join(os.path.dirname(__file__), 'toxcast_data.csv')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(os.path.dirname(__file__), f'toxcast_predictions_{timestamp}.csv')
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return
    
    # Read SMILES (process all compounds)
    smiles_list = DataCreator.read_toxcast_smiles(input_file, max_compounds=30)
    
    if not smiles_list:
        print("❌ No SMILES found in input file!")
        return
    
    # Run predictions
    results = DataCreator.run_predictions(smiles_list)
    
    # Save results
    DataCreator.save_to_csv(results, output_file)
    
    print(f"\n🎉 Prediction pipeline completed successfully!")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main() 
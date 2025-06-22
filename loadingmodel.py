import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os

# --- Step 1: Define the EXACT same model architecture and assay list ---
# This is crucial for loading the saved weights correctly.

class MultiTaskNet(nn.Module):
    """The same neural network architecture used for training."""
    def __init__(self, num_features, num_tasks, dropout_rate=0.5):
        super(MultiTaskNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_tasks)
        )

    def forward(self, x):
        return self.network(x)

def get_full_assay_list():
    """
    Returns the full list of potential assays the model COULD have been trained on.
    This list MUST MATCH the one used during the training of the saved model.
    """
    # This is the original list of 107 assays that resulted in a 96-task model.
    return [
        'ATG_AR_TRANS_dn', 'ATG_AR_TRANS_up', 'ATG_ERE_CIS_dn', 'ATG_ERE_CIS_up', 'ATG_ERa_TRANS_up', 'NVS_NR_bER', 
        'NVS_NR_cAR', 'NVS_NR_hAR', 'NVS_NR_hER', 'NVS_NR_mERa', 'NVS_NR_rAR', 'OT_AR_ARELUC_AG_1440', 
        'OT_AR_ARSRC1_0480', 'OT_AR_ARSRC1_0960', 'OT_ER_ERaERa_0480', 'OT_ER_ERaERa_1440', 'OT_ER_ERaERb_0480', 
        'OT_ER_ERaERb_1440', 'OT_ER_ERbERb_0480', 'OT_ER_ERbERb_1440', 'OT_ERa_EREGFP_0120', 'OT_ERa_EREGFP_0480', 
        'TOX21_AR_BLA_Agonist_ch1', 'TOX21_AR_BLA_Agonist_ch2', 'TOX21_AR_BLA_Agonist_ratio', 'TOX21_AR_BLA_Antagonist_ch1', 
        'TOX21_AR_BLA_Antagonist_ch2', 'TOX21_AR_BLA_Antagonist_ratio', 'TOX21_AR_BLA_Antagonist_viability', 
        'TOX21_AR_LUC_MDAKB2_Agonist', 'TOX21_AR_LUC_MDAKB2_Antagonist', 'TOX21_AR_LUC_MDAKB2_Antagonist2', 
        'TOX21_Aromatase_Inhibition', 'TOX21_ERa_BLA_Agonist_ch1', 'TOX21_ERa_BLA_Agonist_ch2', 
        'TOX21_ERa_BLA_Agonist_ratio', 'TOX21_ERa_BLA_Antagonist_ch1', 'TOX21_ERa_BLA_Antagonist_ch2', 
        'TOX21_ERa_BLA_Antagonist_ratio', 'TOX21_ERa_BLA_Antagonist_viability', 'TOX21_ERa_LUC_BG1_Agonist', 
        'TOX21_ERa_LUC_BG1_Antagonist', 'APR_HepG2_p53Act_24h_up', 'APR_HepG2_p53Act_72h_up', 'APR_Hepat_DNADamage_24hr_up', 
        'APR_Hepat_DNADamage_48hr_up', 'ATG_p53_CIS_up', 'TOX21_p53_BLA_p1_ratio', 'TOX21_p53_BLA_p2_ratio', 
        'TOX21_p53_BLA_p3_ratio', 'TOX21_p53_BLA_p4_ratio', 'TOX21_p53_BLA_p5_ratio',
        'CLD_CYP1A1_24hr', 'CLD_CYP1A2_48hr', 'CLD_CYP2B6_48hr', 'CLD_CYP3A4_48hr', 'NVS_ADME_hCYP19A1', 
        'NVS_ADME_hCYP1A1', 'NVS_ADME_hCYP1A2', 'NVS_ADME_hCYP2A6', 'NVS_ADME_hCYP2B6', 'NVS_ADME_hCYP2C19', 
        'NVS_ADME_hCYP2C9', 'NVS_ADME_hCYP2D6', 'NVS_ADME_hCYP3A4', 'NVS_ADME_hCYP4F12', 'NVS_ADME_rCYP2C12', 
        'NVS_NR_hPXR', 'ATG_PXRE_CIS_up', 'NVS_IC_hKhERGCh', 'ATG_NRF2_ARE_CIS_up', 'TOX21_ARE_BLA_Agonist_ratio', 
        'TOX21_ARE_BLA_agonist_viability', 'ATG_HSE_CIS_up', 'TOX21_HSE_BLA_agonist_ch1', 'TOX21_HSE_BLA_agonist_ch2',
        'TOX21_HSE_BLA_agonist_ratio', 'TOX21_HSE_BLA_agonist_viability',
        'TOX21_NFkB_BLA_agonist_ratio', 'TOX21_p53_BLA_p1_ch1', 'TOX21_p53_BLA_p1_ch2', 'TOX21_p53_BLA_p1_viability',
        'TOX21_p53_BLA_p2_ch1', 'TOX21_p53_BLA_p2_ch2', 'TOX21_p53_BLA_p2_viability', 'TOX21_p53_BLA_p3_ch1',
        'TOX21_p53_BLA_p3_ch2', 'TOX21_p53_BLA_p3_viability', 'TOX21_p53_BLA_p4_ch1', 'TOX21_p53_BLA_p4_ch2',
        'TOX21_p53_BLA_p4_viability', 'TOX21_p53_BLA_p5_ch1', 'TOX21_p53_BLA_p5_ch2', 'TOX21_p53_BLA_p5_viability',
        'APR_Hepat_DNATexture_24hr_up', 'APR_Hepat_DNATexture_48hr_up', 'ATG_E2F_CIS_up'
    ]

# --- Step 2: Create the Prediction Function ---

def predict_toxicity(model, features_list, device, assay_names):
    """Takes a list of feature values and returns predictions from the model."""
    model.eval()  # Set the model to evaluation mode
    
    with torch.no_grad():
        features_tensor = torch.tensor(features_list, dtype=torch.float32).unsqueeze(0).to(device)
        logits = model(features_tensor)
        probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()
        
    results_df = pd.DataFrame({
        'Assay': assay_names,
        'Predicted_Probability_of_Activity': probabilities
    })
    
    return results_df

# --- Step 3: Main execution block to load the model and make predictions ---

if __name__ == '__main__':
    # --- Configuration ---
    MODEL_PATH = 'saved_models/best_toxicology_model0.0597967654466629.pth' 
    FEATURES_CSV_PATH = 'features.csv' # Needed to determine NUM_FEATURES
    ASSAY_CSV_PATH = 'assays.csv' # Needed to determine the exact assay list
    
    # --- Load Data to Determine Model Shape ---
    print("--- Loading data files to determine model architecture ---")
    try:
        # We only need the headers and shape, not the full data
        features_df_full = pd.read_csv(FEATURES_CSV_PATH)
        assays_df_full = pd.read_csv(ASSAY_CSV_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure '{ASSAY_CSV_PATH}' and '{FEATURES_CSV_PATH}' are in the project directory.")
        exit()

    # Dynamically determine the exact number of features and tasks
    NUM_FEATURES = len(features_df_full.columns)
    
    # Get the precise list of assays the model was trained on
    full_assay_list = get_full_assay_list()
    assays_in_df = [assay for assay in full_assay_list if assay in assays_df_full.columns]
    NUM_TASKS = len(assays_in_df)

    print(f"Inferred architecture from data files:")
    print(f" - Input Features (NUM_FEATURES): {NUM_FEATURES}")
    print(f" - Output Assays (NUM_TASKS): {NUM_TASKS}")


    # --- Load the Model ---
    print(f"\n--- Loading model from '{MODEL_PATH}' ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Instantiate the model architecture with the dynamically determined shape
    prediction_model = MultiTaskNet(num_features=NUM_FEATURES, num_tasks=NUM_TASKS).to(device)
    
    try:
        prediction_model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(device)))
        print("Model loaded successfully!")
    except FileNotFoundError:
        print(f"Error: Model file not found at '{MODEL_PATH}'.")
        print("Please check the path and filename.")
        exit()
    except RuntimeError as e:
        print(f"A runtime error occurred while loading the model: {e}")
        print("\nThis usually means the NUM_FEATURES or NUM_TASKS in this script does not match the saved model.")
        print("Please ensure the 'features.csv' and 'assays.csv' files are the same ones used for training.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        exit()

    # --- Make a Prediction on New Data ---
    
    # Example: Create a new data point to test the model.
    # Replace this with the actual features of the compound you want to predict.
    example_features = np.random.rand(NUM_FEATURES).tolist()
    
    print(f"\n--- Predicting toxicity for a new compound ---")
    print(f"Input Features (first 5): {example_features[:5]}...")
    
    # Use the dynamically generated list of assay names for labeling
    predictions = predict_toxicity(prediction_model, example_features, device, assays_in_df)

    # --- Display the Results ---
    print("\n--- Prediction Results ---")
    sorted_predictions = predictions.sort_values(by='Predicted_Probability_of_Activity', ascending=False)
    
    print("\nTop 15 most likely activities:")
    print(sorted_predictions.head(15).to_string())
    
    print("\n\nTop 15 least likely activities:")
    print(sorted_predictions.tail(15).to_string())

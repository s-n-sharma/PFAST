import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import copy
import sys
import os
from typing import List
from run_toxcast_predictions import DataCreator

# Add the parent directory to the path to import the predictors
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pfas_prediction_models.combined_pfas import CombinedPFASPredictor


class ToxAssayDataset(Dataset):
    """Custom PyTorch Dataset to handle pre-loaded feature and label dataframes."""
    def __init__(self, features_df, ignore_value=-255):
        self.features = torch.tensor(features_df.values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

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
def run_model(smile : str):
    """
    Run the model on a single SMILES string.
    """
    # Get feature vector
    features = DataCreator.get_feature_vector(smile)
    features_df = pd.DataFrame([features])
    features_df.columns = ['feature_' + str(i) for i in range(features_df.shape[1])]
    
    # Load the model
    try:
        # Load state dict
        state_dict = torch.load('bestmodel.pth', map_location=torch.device('cpu'))
        
        # Determine model architecture from state dict
        first_layer_weight = state_dict['network.0.weight']
        last_layer_weight = state_dict['network.11.weight']
        
        num_features = first_layer_weight.shape[1]
        num_tasks = last_layer_weight.shape[0]
        
        # Create model with correct architecture
        model = MultiTaskNet(num_features, num_tasks)
        model.load_state_dict(state_dict)
        model.eval()
        
        print(f"✓ Model loaded: {num_features} features -> {num_tasks} tasks")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create dataset and dataloader
    dataset = ToxAssayDataset(features_df)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Run prediction
    with torch.no_grad():
        for features in dataloader:
            predictions = model(features)
            probabilities = torch.sigmoid(predictions)
            return probabilities[0].tolist()

def get_important_assays():
    """Returns the curated list of 107 important assay names."""
    # This list is based on our previous discussion, prioritizing key toxicity pathways.
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
    
def prob_and_assay_name(probabilities : List[float]):
    """
    Returns the probability and assay name for the highest probability.
    """
    assay_prob = {}
    for i, prob in enumerate(probabilities):
        assay_prob[get_important_assays()[i]] = prob
    return assay_prob
    
if __name__ == "__main__":
    # Test with a PFAS compound
    test_smiles = "C(=O)(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O"  # PFOA
    print(f"Testing with SMILES: {test_smiles}")
    result = run_model(test_smiles)
    dictionary = prob_and_assay_name(result)
    print(dictionary)
   
        

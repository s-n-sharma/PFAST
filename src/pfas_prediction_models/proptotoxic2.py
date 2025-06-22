import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import copy

# Note: RDKit is no longer required for this script as features are pre-calculated.

# --- Part A: Data Loading and Preprocessing ---

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

# --- Part B: Neural Network and Custom Loss Function ---

class ToxAssayDataset(Dataset):
    """Custom PyTorch Dataset to handle pre-loaded feature and label dataframes."""
    def __init__(self, features_df, labels_df, ignore_value=-255):
        aligned_labels_df = labels_df.loc[features_df.index]
        self.features = torch.tensor(features_df.values, dtype=torch.float32)
        self.labels = torch.tensor(aligned_labels_df.fillna(ignore_value).values, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

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

def masked_bce_loss(output, target, ignore_value=-255):
    """Custom loss function that ignores a specific value in the target tensor."""
    mask = target != ignore_value
    if not torch.any(mask):
        return torch.tensor(0.0, device=output.device, requires_grad=True)
    
    output_masked = output[mask]
    target_masked = target[mask]
    
    loss_fn = nn.BCEWithLogitsLoss()
    return loss_fn(output_masked, target_masked)

# --- Part C: Evaluation and Training Functions ---

def evaluate_model(model, data_loader, device):
    """Evaluates the model on a given dataset (e.g., validation or test set)."""
    model.eval()
    total_loss = 0
    criterion = masked_bce_loss

    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    
    return total_loss / len(data_loader) if len(data_loader) > 0 else 0

# --- Main Execution Block ---
if __name__ == '__main__':
    # --- Configuration ---
    ASSAY_CSV_PATH = 'assays.csv'
    FEATURES_CSV_PATH = 'features.csv'
    
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3 # Increased LR, as BatchNorm allows for it
    EPOCHS = 100 # Increased epochs for LR scheduler and early stopping
    WEIGHT_DECAY = 5e-5 # Slightly increased L2 regularization
    EARLY_STOPPING_PATIENCE = 10 # Stop after 10 epochs of no improvement

    # A. Load and preprocess data
    print("--- 1. Data Loading and Preprocessing ---")
    try:
        assays_df_full = pd.read_csv(ASSAY_CSV_PATH)
        features_df_full = pd.read_csv(FEATURES_CSV_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure both '{ASSAY_CSV_PATH}' and '{FEATURES_CSV_PATH}' are in the correct directory.")
        exit()
    
    assert len(assays_df_full) == len(features_df_full), "Assay and Feature CSVs must have the same number of rows."

    smiles_col = 'SMILES' if 'SMILES' in assays_df_full.columns else assays_df_full.columns[0]
    print(f"Identified SMILES column as: '{smiles_col}'")
    
    print("Cleaning features data...")
    for col in features_df_full.columns:
        features_df_full[col] = pd.to_numeric(features_df_full[col], errors='coerce')
    features_df_full.fillna(0, inplace=True)

    NUM_FEATURES = len(features_df_full.columns)
    print(f"Dynamically detected {NUM_FEATURES} input features.")

    important_assays = get_important_assays()
    assays_in_df = [assay for assay in important_assays if assay in assays_df_full.columns]
    NUM_TASKS = len(assays_in_df)
    
    if NUM_TASKS == 0:
        print("Error: No important assays found in the provided assay.csv. Please check column names.")
        exit()
        
    print(f"Found {NUM_TASKS} of the important assays in '{ASSAY_CSV_PATH}'.")

    labels_df = assays_df_full[assays_in_df].copy()
    for col in assays_in_df:
        labels_df[col] = pd.to_numeric(labels_df[col], errors='coerce')

    # C. Create Train, Validation, and Test Splits
    print("\n--- 2. Creating Train/Validation/Test Splits ---")
    smiles_series = assays_df_full[smiles_col].dropna()
    
    contains_f_indices = smiles_series[smiles_series.str.contains('F', na=False)].index
    no_f_indices = smiles_series[~smiles_series.str.contains('F', na=False)].index

    f_sample_size = min(5, len(contains_f_indices))
    no_f_sample_size = min(5, len(no_f_indices))

    test_indices = pd.Index([])
    if f_sample_size > 0:
        test_indices = test_indices.union(pd.Index(np.random.choice(contains_f_indices, size=f_sample_size, replace=False)))
    if no_f_sample_size > 0:
        available_no_f = no_f_indices.difference(test_indices)
        test_indices = test_indices.union(pd.Index(np.random.choice(available_no_f, size=min(no_f_sample_size, len(available_no_f)), replace=False)))
    
    temp_train_indices = assays_df_full.index.difference(test_indices)
    validation_size = int(len(temp_train_indices) * 0.15)
    train_indices = pd.Index(np.random.choice(temp_train_indices, size=len(temp_train_indices) - validation_size, replace=False))
    validation_indices = temp_train_indices.difference(train_indices)

    X_train, y_train = features_df_full.loc[train_indices], labels_df.loc[train_indices]
    X_val, y_val = features_df_full.loc[validation_indices], labels_df.loc[validation_indices]
    X_test, y_test = features_df_full.loc[test_indices], labels_df.loc[test_indices]

    print(f"Total data: {len(assays_df_full)} compounds.")
    print(f"Training set size: {len(X_train)} compounds.")
    print(f"Validation set size: {len(X_val)} compounds.")
    print(f"Test set size: {len(X_test)} compounds ({f_sample_size} with 'F', {no_f_sample_size} without 'F').")

    train_dataset = ToxAssayDataset(X_train, y_train)
    val_dataset = ToxAssayDataset(X_val, y_val)
    test_dataset = ToxAssayDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) if len(test_dataset) > 0 else None

    # B. Initialize Model and start training
    print("\n--- 3. Model Training with Advanced Techniques ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiTaskNet(num_features=NUM_FEATURES, num_tasks=NUM_TASKS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    criterion = masked_bce_loss
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0
        avg_val_loss = evaluate_model(model, val_loader, device)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        scheduler.step(avg_val_loss) # Adjust learning rate based on validation loss
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"New best validation loss: {best_val_loss:.6f}. Saving model.")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs with no improvement.")
            break

    print("\nTraining finished.")
    
    if best_model_state:
        model.load_state_dict(best_model_state)
        print("Loaded best model state for final evaluation.")

    if test_loader:
        print("\n--- 4. Final Evaluation on Unseen Test Set ---")
        test_loss = evaluate_model(model, test_loader, device)
        print(f"Final Test Loss: {test_loss:.6f}")
    else:
        print("\n--- 4. Final Evaluation on Unseen Test Set ---")
        print("Test set is empty, skipping final evaluation.")
    
    
    if best_model_state:
        save_choice = input("\nDo you want to save the best model to a file? (y/n): ").lower()
        if save_choice in ['y', 'yes']:
            MODEL_SAVE_PATH = f'saved_models/best_toxicology_model{test_loss}.pth'
            try:
                torch.save(best_model_state, MODEL_SAVE_PATH)
                print(f"Model successfully saved to '{MODEL_SAVE_PATH}'")
            except Exception as e:
                print(f"Error saving model: {e}")
        else:
            print("Model not saved.")

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import io

# --- 1. DATASET GENERATION ---
# In a real-world scenario, you would load this from a CSV file.
# Here, we generate a synthetic dataset for demonstration.
# The 'Risk_Score' (1-10) is a simplified metric derived from EPA health advisory levels (HAL).
pfas_hals = {
    'PFOA': 4.0,
    'PFOS': 4.0,
    'GenX': 10.0,
    'PFBS': 2000.0,
    'PFHxS': 10.0,
    'PFNA': 10.0
}

def concentration_to_risk(concentration, hal):
    """Converts concentration to a 1-10 risk score based on the Health Advisory Level (HAL)."""
    if concentration < hal * 0.1:
        return 1
    if concentration < hal * 0.5:
        return 2
    if concentration < hal:
        return 4
    if concentration < hal * 5:
        return 6
    if concentration < hal * 10:
        return 8
    if concentration < hal * 50:
        return 9
    return 10

data_list = []
for chemical, hal in pfas_hals.items():
    for _ in range(200): # Generate 200 samples per chemical
        # Generate concentrations across a wide, logarithmic range relative to the HAL
        concentration = np.random.power(0.5) * hal * 100
        risk = concentration_to_risk(concentration, hal)
        data_list.append([chemical, concentration, risk])

# Create a pandas DataFrame
df = pd.DataFrame(data_list, columns=['Chemical', 'Concentration_ppt', 'Risk_Score'])

print("--- Generated Dataset Sample ---")
print(df.head())
print("\n")


# --- 2. DATA PREPROCESSING ---

# Define features (X) and target (y)
X = df[['Chemical', 'Concentration_ppt']]
y = df[['Risk_Score']]

# Create preprocessing pipelines for numerical and categorical features
# We scale numerical features and one-hot encode categorical features.
numerical_features = ['Concentration_ppt']
categorical_features = ['Chemical']

# Pipeline for numerical data: standard scaling
numerical_transformer = StandardScaler()

# Pipeline for categorical data: one-hot encoding
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Combine preprocessing steps using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply the preprocessing pipeline to the data
# We fit on the training data and transform both train and test data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Convert to PyTorch Tensors
X_train_tensor = torch.tensor(X_train_processed.toarray(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_processed.toarray(), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Create TensorDatasets and DataLoaders for batching
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)


# --- 3. MODEL DEFINITION (MLP) ---

class PFAS_Risk_MLP(nn.Module):
    def __init__(self, input_size):
        super(PFAS_Risk_MLP, self).__init__()
        # Define the layers of the neural network
        self.layer1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.output_layer = nn.Linear(32, 1) # Output is a single risk score

    def forward(self, x):
        # Define the forward pass
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.output_layer(x)
        return x

# Instantiate the model
input_dim = X_train_processed.shape[1]
model = PFAS_Risk_MLP(input_dim)
print("--- Model Architecture ---")
print(model)
print("\n")


# --- 4. TRAINING THE MODEL ---

# Loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
print("--- Starting Model Training ---")

for epoch in range(num_epochs):
    model.train() # Set the model to training mode
    epoch_loss = 0.0
    for inputs, targets in train_loader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}')
print("--- Training Finished ---\n")


# --- 5. EVALUATION AND PREDICTION ---

model.eval() # Set the model to evaluation mode
with torch.no_grad(): # No need to calculate gradients for evaluation
    # Evaluate on test data
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor)
    print(f'Test Mean Squared Error: {test_loss.item():.4f}')

    # Show some example predictions
    print("\n--- Example Predictions on Test Data ---")
    for i in range(5):
        original_input = X_test.iloc[i]
        predicted_risk = test_outputs[i].item()
        actual_risk = y_test_tensor[i].item()

        print(f"Input: Chemical={original_input['Chemical']}, Concentration={original_input['Concentration_ppt']:.2f} ppt")
        print(f"  -> Predicted Risk: {predicted_risk:.2f}, Actual Risk: {actual_risk}")

# --- PREDICTION FUNCTION ---
def predict_risk(chemical_name, concentration_ppt):
    """Predicts the risk for a single new data point."""
    model.eval()
    # Create a DataFrame for the new data point to use the same preprocessor
    new_data = pd.DataFrame([[chemical_name, concentration_ppt]], columns=['Chemical', 'Concentration_ppt'])
    
    # Preprocess the new data
    new_data_processed = preprocessor.transform(new_data)
    
    # Convert to tensor
    new_data_tensor = torch.tensor(new_data_processed.toarray(), dtype=torch.float32)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(new_data_tensor)
        # Clamp the output between 1 and 10
        clamped_prediction = torch.clamp(prediction, 1, 10).item()

    return clamped_prediction

print("\n--- Live Prediction Example ---")
# Example: Predict the risk for PFOA at 15 ppt
new_chemical = 'PFOA'
new_concentration = 15.0
predicted_score = predict_risk(new_chemical, new_concentration)
print(f"Prediction for {new_chemical} at {new_concentration} ppt -> Risk Score: {predicted_score:.2f}")

# Example: Predict the risk for PFBS at 1500 ppt
new_chemical = 'PFBS'
new_concentration = 1500.0
predicted_score = predict_risk(new_chemical, new_concentration)
print(f"Prediction for {new_chemical} at {new_concentration} ppt -> Risk Score: {predicted_score:.2f}")


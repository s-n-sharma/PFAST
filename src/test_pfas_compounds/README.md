# PFAS Property Prediction Pipeline

This folder contains a complete pipeline for predicting properties of PFAS compounds using the combined predictor model.

## Files

- `pfas_smiles_input.txt` - Input file containing PFAS SMILES strings
- `run_pfas_predictions.py` - Python script to run predictions and save results
- `README.md` - This file

## Usage

### 1. Prepare Input Data

Edit `pfas_smiles_input.txt` to include your PFAS compounds:

```
# Format: Compound_Name: SMILES_String
PFOA: C(=O)(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O
PFOS: C(C(C(C(C(F)(F)S(=O)(=O)O)(F)F)(F)F)(F)F)(C(C(C(F)(F)F)(F)F)(F)F)(F)F
```

### 2. Run Predictions

From the `src` directory, run:

```bash
python test_pfas_compounds/run_pfas_predictions.py
```

### 3. View Results

The script will create a timestamped CSV file (e.g., `pfas_predictions_20241221_143022.csv`) containing:

- Compound information (name, SMILES)
- Summary statistics (total properties, method counts)
- All predicted properties with values, units, and methods used

## Output Format

The CSV file contains columns for each property:
- `{property_name}_value` - Predicted value
- `{property_name}_unit` - Unit of measurement
- `{property_name}_method` - Method used (moleculenet/accurate)

## Available Properties

The combined predictor provides predictions for:

**MoleculeNet Properties:**
- Solubility (logS)
- Lipophilicity (logP)
- Blood-brain barrier penetration (BBBP)
- HIV replication inhibition

**RDKit Properties:**
- Molecular weight, formula
- LogP, LogD, TPSA
- Hydrogen bond donors/acceptors
- Rotatable bonds, ring counts
- Atom composition (C, H, O, N, S, F)
- Advanced structural properties
- Drug-likeness and toxicity scores

## Notes

- Models are automatically loaded from saved files for fast startup
- Failed predictions are logged with error messages
- Results include both successful and failed predictions for transparency 
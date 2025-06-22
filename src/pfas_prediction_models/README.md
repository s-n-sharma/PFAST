# PFAS Property Prediction Models

A modular, extensible system for predicting properties of PFAS compounds using RDKit, MoleculeNet (DeepChem), and combined approaches.

## Structure

- `accurate_pfas.py`: Accurate RDKit-based property predictor (baseline)
- `moleculenet_pfas.py`: MoleculeNet/DeepChem-based property predictor
- `combined_pfas.py`: Combines RDKit, MoleculeNet, and numerical/statistical features
- `utils.py`: Shared utilities (e.g., SMILES validation)
- `test/`: Unit tests for each model

## Usage

```
from pfas_prediction_models.accurate_pfas import AccuratePFASPredictor
from pfas_prediction_models.moleculenet_pfas import MoleculeNetPFASPredictor
from pfas_prediction_models.combined_pfas import CombinedPFASPredictor

smiles = "C(=O)(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O"  # PFOA

# Accurate RDKit-based prediction
acc = AccuratePFASPredictor()
print(acc.predict_all_properties(smiles))

# MoleculeNet/DeepChem-based prediction
molnet = MoleculeNetPFASPredictor()
print(molnet.predict(smiles))

# Combined prediction
comb = CombinedPFASPredictor()
print(comb.predict_all(smiles))
```

## Testing

Run tests from the `src/pfas_prediction_models/test/` directory:

```
python test_accurate.py
python test_moleculenet.py
python test_combined.py
```

## Extending
- Add new models or features by creating new modules and updating `combined_pfas.py`.
- Use `utils.py` for shared code.

## Requirements
- RDKit
- DeepChem
- scikit-learn
- numpy, pandas

---

For more details, see each module's docstring. 
"""
Utilities for PFAS Prediction Models
"""

from rdkit import Chem

def is_valid_smiles(smiles: str) -> bool:
    """Check if a SMILES string is valid."""
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None 
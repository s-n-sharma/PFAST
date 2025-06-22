"""
PFAS Property Prediction Models Package

A modular, extensible system for predicting properties of PFAS compounds.
"""

# Import main classes for easy access
from .accurate_pfas import AccuratePFASPredictor
from .moleculenet_pfas import MoleculeNetPFASPredictor
from .combined_pfas import CombinedPFASPredictor
from .utils import is_valid_smiles

# Version info
__version__ = "1.0.0"

# Package exports
__all__ = [
    'AccuratePFASPredictor',
    'MoleculeNetPFASPredictor', 
    'CombinedPFASPredictor',
    'is_valid_smiles'
] 
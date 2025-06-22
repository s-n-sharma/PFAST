from pfas_prediction_models.accurate_pfas import AccuratePFASPredictor

def test_accurate_pfas_predictor():
    predictor = AccuratePFASPredictor()
    smiles = "C(=O)(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O"  # PFOA
    result = predictor.predict_all_properties(smiles)
    assert result is not None
    assert 'molecular_weight' in result['properties']
    assert result['properties']['molecular_weight']['value'] > 0
    print("AccuratePFASPredictor test passed.")

if __name__ == "__main__":
    test_accurate_pfas_predictor() 
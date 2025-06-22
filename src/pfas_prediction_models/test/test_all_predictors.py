from pfas_prediction_models.accurate_pfas import AccuratePFASPredictor
from pfas_prediction_models.moleculenet_pfas import MoleculeNetPFASPredictor
from pfas_prediction_models.combined_pfas import CombinedPFASPredictor

PFAS_SMILES = {
    "PFOA": "C(=O)(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O",
    "PFOS": "C(C(C(C(C(F)(F)S(=O)(=O)O)(F)F)(F)F)(F)F)(C(C(C(F)(F)F)(F)F)(F)F)(F)F",
    "GenX": "C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(C(=O)O)C(=O)O",
}


def print_property_table(results_dict, model_name):
    print(f"\n{'='*30}\n{model_name} Results\n{'='*30}")
    for name, result in results_dict.items():
        print(f"\n{name} ({result['smiles']}):")
        if 'properties' in result:
            for prop, data in result['properties'].items():
                value = data['value']
                if isinstance(value, (int, float)):
                    print(f"  {prop}: {value:.4f} {data['unit']}")
                else:
                    print(f"  {prop}: {value} {data['unit']}")
        else:
            print("  No properties predicted.")

def compare_models(acc_results, molnet_results, comb_results):
    print(f"\n{'='*30}\nMODEL COMPARISON\n{'='*30}")
    for name in PFAS_SMILES:
        print(f"\n{name}:")
        acc_props = acc_results[name]['properties']
        molnet_props = molnet_results[name]['properties']
        comb_props = comb_results[name]['properties']
        # Find overlapping properties
        overlap = set(acc_props.keys()) & set(molnet_props.keys())
        if not overlap:
            print("  No overlapping properties.")
            continue
        for prop in overlap:
            acc_val = acc_props[prop]['value']
            molnet_val = molnet_props[prop]['value']
            comb_val = comb_props[prop]['value'] if prop in comb_props else None
            print(f"  {prop}:")
            print(f"    Accurate:    {acc_val}")
            print(f"    MoleculeNet: {molnet_val}")
            print(f"    Combined:    {comb_val}")
            # Optionally, show difference if numeric
            if isinstance(acc_val, (int, float)) and isinstance(molnet_val, (int, float)):
                print(f"    |Accurate - MoleculeNet|: {abs(acc_val - molnet_val):.4f}")
            if comb_val is not None and isinstance(acc_val, (int, float)) and isinstance(comb_val, (int, float)):
                print(f"    |Accurate - Combined|: {abs(acc_val - comb_val):.4f}")
            if comb_val is not None and isinstance(molnet_val, (int, float)) and isinstance(comb_val, (int, float)):
                print(f"    |MoleculeNet - Combined|: {abs(molnet_val - comb_val):.4f}")

def main():
    print("Testing all PFAS predictors and comparing results...")
    # Initialize predictors (this will also save models if not already saved)
    acc = AccuratePFASPredictor()
    molnet = MoleculeNetPFASPredictor()
    comb = CombinedPFASPredictor()
    # Run predictions
    acc_results = {}
    molnet_results = {}
    comb_results = {}
    for name, smiles in PFAS_SMILES.items():
        acc_results[name] = acc.predict_all_properties(smiles)
        molnet_results[name] = molnet.predict_all_properties(smiles)
        comb_results[name] = comb.predict_all(smiles)
    # Print results
    print_property_table(acc_results, "AccuratePFASPredictor")
    print_property_table(molnet_results, "MoleculeNetPFASPredictor")
    print_property_table(comb_results, "CombinedPFASPredictor")
    # Compare
    compare_models(acc_results, molnet_results, comb_results)
    print("\nAll predictor tests and comparisons complete.")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
import pickle
import numpy as np
import sys

def load_pickle(filepath):
    """Load pickle file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def exclude_s2_h3_features(sample):
    """Remove S2 and H3 features from a sample"""
    if not isinstance(sample, dict):
        return sample
    
    # Create a copy without modifying the original
    filtered_sample = {}
    for key, value in sample.items():
        # Skip keys that contain 's2' or 'h3' (case insensitive)
        key_lower = key.lower()
        if 's2' in key_lower or 'h3' in key_lower:
            continue
        filtered_sample[key] = value
    
    return filtered_sample

def compare_nested_structure(obj1, obj2):
    """Recursively compare nested structures"""
    if type(obj1) != type(obj2):
        return False
    
    if isinstance(obj1, np.ndarray):
        return np.array_equal(obj1, obj2)
    elif isinstance(obj1, (list, tuple)):
        if len(obj1) != len(obj2):
            return False
        return all(compare_nested_structure(e1, e2) for e1, e2 in zip(obj1, obj2))
    elif isinstance(obj1, dict):
        if set(obj1.keys()) != set(obj2.keys()):
            return False
        return all(compare_nested_structure(obj1[k], obj2[k]) for k in obj1.keys())
    else:
        return obj1 == obj2

def compare_samples(data1, data2, file_pair_name):
    """Compare two datasets sample by sample, excluding S2 and H3 features"""
    print(f"\n{'='*80}")
    print(f"Comparing: {file_pair_name}")
    print(f"{'='*80}")
    
    # Check if both are the same type
    if type(data1) != type(data2):
        print(f"❌ FAIL: Different types - {type(data1)} vs {type(data2)}")
        return False
    
    # Handle list of samples
    if isinstance(data1, (list, tuple)):
        if len(data1) != len(data2):
            print(f"❌ FAIL: Different number of samples - {len(data1)} vs {len(data2)}")
            return False
        
        print(f"Number of samples: {len(data1)}")
        
        # Check structure of first sample
        if len(data1) > 0:
            sample1 = data1[0]
            sample2 = data2[0]
            print(f"Sample structure: {type(sample1)}")
            
            if isinstance(sample1, dict):
                print(f"  Sample 1 keys: {sorted(sample1.keys())}")
                print(f"  Sample 2 keys: {sorted(sample2.keys())}")
                
                # Filter out S2 and H3 features
                filtered1 = exclude_s2_h3_features(sample1)
                filtered2 = exclude_s2_h3_features(sample2)
                
                print(f"\n  After excluding S2/H3 features:")
                print(f"  Filtered sample 1 keys: {sorted(filtered1.keys())}")
                print(f"  Filtered sample 2 keys: {sorted(filtered2.keys())}")
                
                # Show details of each key
                for key in sorted(filtered1.keys()):
                    val = filtered1[key]
                    if isinstance(val, np.ndarray):
                        print(f"    {key}: numpy array, shape={val.shape}, dtype={val.dtype}")
                    elif isinstance(val, (list, tuple)):
                        print(f"    {key}: {type(val).__name__}, length={len(val)}")
                    else:
                        print(f"    {key}: {type(val).__name__}")
        
        all_match = True
        mismatches = []
        
        for i in range(len(data1)):
            sample1 = exclude_s2_h3_features(data1[i])
            sample2 = exclude_s2_h3_features(data2[i])
            
            match = compare_nested_structure(sample1, sample2)
            if not match:
                all_match = False
                mismatches.append(i)
                if len(mismatches) <= 5:  # Show first 5 mismatches
                    print(f"❌ Sample {i}: Differs")
                    # Show what differs
                    if isinstance(sample1, dict) and isinstance(sample2, dict):
                        keys1 = set(sample1.keys())
                        keys2 = set(sample2.keys())
                        if keys1 != keys2:
                            print(f"   Different keys: only in 1: {keys1-keys2}, only in 2: {keys2-keys1}")
                        else:
                            for key in sample1.keys():
                                if not compare_nested_structure(sample1[key], sample2[key]):
                                    val1 = sample1[key]
                                    val2 = sample2[key]
                                    if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
                                        if val1.shape != val2.shape:
                                            print(f"   Key '{key}': different shapes {val1.shape} vs {val2.shape}")
                                        else:
                                            diff = np.abs(val1 - val2)
                                            print(f"   Key '{key}': arrays differ, max diff={np.max(diff):.6f}")
                                    else:
                                        print(f"   Key '{key}': values differ")
        
        if all_match:
            print(f"\n✓ ALL {len(data1)} SAMPLES IDENTICAL (excluding S2/H3 features)")
        else:
            print(f"\n❌ DATASETS DIFFER: {len(mismatches)} out of {len(data1)} samples differ")
            if len(mismatches) > 5:
                print(f"   (showing first 5 mismatches only)")
        
        return all_match
    
    else:
        print(f"❌ Unexpected data structure: {type(data1)}")
        return False

def main():
    # Define file pairs to compare
    file_pairs = [
        ("train", 
         "/data/next_loc_clean_v2/data/geolife_eps20/processed/geolife_eps20_prev7_train.pk",
         "/data/location-prediction-ori-freeze/data/temp/geolife_transformer_7_train.pk"),
        ("validation",
         "/data/next_loc_clean_v2/data/geolife_eps20/processed/geolife_eps20_prev7_validation.pk",
         "/data/location-prediction-ori-freeze/data/temp/geolife_transformer_7_validation.pk"),
        ("test",
         "/data/next_loc_clean_v2/data/geolife_eps20/processed/geolife_eps20_prev7_test.pk",
         "/data/location-prediction-ori-freeze/data/temp/geolife_transformer_7_test.pk"),
    ]
    
    all_files_match = True
    
    for name, file1, file2 in file_pairs:
        try:
            print(f"\nLoading {name} datasets...")
            print(f"  Dataset 1: {file1}")
            print(f"  Dataset 2: {file2}")
            
            data1 = load_pickle(file1)
            data2 = load_pickle(file2)
            
            match = compare_samples(data1, data2, name)
            if not match:
                all_files_match = False
                
        except Exception as e:
            print(f"\n❌ ERROR comparing {name}: {e}")
            import traceback
            traceback.print_exc()
            all_files_match = False
    
    print(f"\n{'='*80}")
    print("FINAL RESULT")
    print(f"{'='*80}")
    if all_files_match:
        print("✓ ALL DATASETS ARE EXACTLY THE SAME (excluding S2/H3 features)")
        sys.exit(0)
    else:
        print("❌ DATASETS DIFFER")
        sys.exit(1)

if __name__ == "__main__":
    main()

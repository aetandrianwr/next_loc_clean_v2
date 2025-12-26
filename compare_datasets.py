#!/usr/bin/env python3
import pickle
import numpy as np
import sys

def load_pickle(filepath):
    """Load pickle file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def compare_samples(data1, data2, file_pair_name):
    """Compare two datasets sample by sample"""
    print(f"\n{'='*80}")
    print(f"Comparing: {file_pair_name}")
    print(f"{'='*80}")
    
    # Check if both are the same type
    if type(data1) != type(data2):
        print(f"❌ FAIL: Different types - {type(data1)} vs {type(data2)}")
        return False
    
    # Handle different data structures
    if isinstance(data1, dict):
        return compare_dict_samples(data1, data2, file_pair_name)
    elif isinstance(data1, (list, tuple)):
        return compare_list_samples(data1, data2, file_pair_name)
    elif isinstance(data1, np.ndarray):
        return compare_array_samples(data1, data2, file_pair_name)
    else:
        # Direct comparison
        if data1 == data2:
            print(f"✓ Datasets are identical")
            return True
        else:
            print(f"❌ FAIL: Datasets differ")
            return False

def compare_dict_samples(data1, data2, file_pair_name):
    """Compare dictionary datasets"""
    keys1 = set(data1.keys())
    keys2 = set(data2.keys())
    
    if keys1 != keys2:
        print(f"❌ FAIL: Different keys")
        print(f"  Only in dataset1: {keys1 - keys2}")
        print(f"  Only in dataset2: {keys2 - keys1}")
        return False
    
    print(f"Keys: {sorted(keys1)}")
    all_match = True
    
    for key in sorted(keys1):
        val1 = data1[key]
        val2 = data2[key]
        
        if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            if val1.shape != val2.shape:
                print(f"❌ Key '{key}': Different shapes - {val1.shape} vs {val2.shape}")
                all_match = False
            elif not np.array_equal(val1, val2):
                print(f"❌ Key '{key}': Arrays differ (shape: {val1.shape})")
                # Show some statistics about the difference
                if val1.dtype in [np.float32, np.float64]:
                    diff = np.abs(val1 - val2)
                    print(f"  Max absolute difference: {np.max(diff)}")
                    print(f"  Mean absolute difference: {np.mean(diff)}")
                all_match = False
            else:
                print(f"✓ Key '{key}': Identical (shape: {val1.shape}, dtype: {val1.dtype})")
        elif isinstance(val1, (list, tuple)) and isinstance(val2, (list, tuple)):
            if len(val1) != len(val2):
                print(f"❌ Key '{key}': Different lengths - {len(val1)} vs {len(val2)}")
                all_match = False
            else:
                match = all(v1 == v2 for v1, v2 in zip(val1, val2))
                if match:
                    print(f"✓ Key '{key}': Identical (length: {len(val1)})")
                else:
                    print(f"❌ Key '{key}': Lists differ (length: {len(val1)})")
                    all_match = False
        else:
            if val1 == val2:
                print(f"✓ Key '{key}': Identical")
            else:
                print(f"❌ Key '{key}': Values differ")
                all_match = False
    
    if all_match:
        print(f"\n✓ ALL SAMPLES IDENTICAL")
    else:
        print(f"\n❌ DATASETS DIFFER")
    
    return all_match

def compare_list_samples(data1, data2, file_pair_name):
    """Compare list/tuple datasets"""
    if len(data1) != len(data2):
        print(f"❌ FAIL: Different number of samples - {len(data1)} vs {len(data2)}")
        return False
    
    print(f"Number of samples: {len(data1)}")
    
    # Check structure of first sample
    if len(data1) > 0:
        sample1 = data1[0]
        sample2 = data2[0]
        print(f"Sample structure: {type(sample1)}")
        if isinstance(sample1, (list, tuple)):
            print(f"  Sample contains {len(sample1)} elements")
            for idx, elem in enumerate(sample1):
                if isinstance(elem, np.ndarray):
                    print(f"    Element {idx}: numpy array, shape={elem.shape}, dtype={elem.dtype}")
                else:
                    print(f"    Element {idx}: {type(elem)}")
        elif isinstance(sample1, np.ndarray):
            print(f"  Sample is numpy array: shape={sample1.shape}, dtype={sample1.dtype}")
    
    all_match = True
    mismatches = []
    
    for i in range(len(data1)):
        sample1 = data1[i]
        sample2 = data2[i]
        
        match = compare_nested_structure(sample1, sample2)
        if not match:
            all_match = False
            mismatches.append(i)
            if len(mismatches) <= 5:  # Show first 5 mismatches
                print(f"❌ Sample {i}: Differs")
    
    if all_match:
        print(f"✓ ALL {len(data1)} SAMPLES IDENTICAL")
    else:
        print(f"❌ DATASETS DIFFER: {len(mismatches)} out of {len(data1)} samples differ")
        if len(mismatches) > 5:
            print(f"   (showing first 5 mismatches only)")
    
    return all_match

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

def compare_array_samples(data1, data2, file_pair_name):
    """Compare numpy array datasets"""
    if data1.shape != data2.shape:
        print(f"❌ FAIL: Different shapes - {data1.shape} vs {data2.shape}")
        return False
    
    if not np.array_equal(data1, data2):
        print(f"❌ FAIL: Arrays differ (shape: {data1.shape})")
        if data1.dtype in [np.float32, np.float64]:
            diff = np.abs(data1 - data2)
            print(f"  Max absolute difference: {np.max(diff)}")
            print(f"  Mean absolute difference: {np.mean(diff)}")
        return False
    
    print(f"✓ Arrays identical (shape: {data1.shape}, dtype: {data1.dtype})")
    return True

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
        print("✓ ALL DATASETS ARE EXACTLY THE SAME")
        sys.exit(0)
    else:
        print("❌ DATASETS DIFFER")
        sys.exit(1)

if __name__ == "__main__":
    main()

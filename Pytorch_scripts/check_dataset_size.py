#!/usr/bin/env python3
"""
Quick diagnostic script to check how many files are in your datasets.
"""
import glob
import os

# Update these paths to your actual dataset locations
AIRGUN_DIR = "/Users/oceaneboulais/Github/ThodeLab/BCB_Whale_Datasets/Unsupervised_database_ManyAirguns.dir"
WHALE_DIR = "/Users/oceaneboulais/Github/ThodeLab/BCB_Whale_Datasets/Unsupervised_database_ManyWhaleCalls.dir"

def count_mat_files(directory):
    """Count .mat files recursively in a directory."""
    if not os.path.exists(directory):
        print(f"ERROR: Directory not found: {directory}")
        return 0
    
    mat_files = glob.glob(os.path.join(directory, '**', '*.mat'), recursive=True)
    return len(mat_files)

if __name__ == "__main__":
    print("="*60)
    print("Dataset File Count Diagnostic")
    print("="*60)
    
    airgun_count = count_mat_files(AIRGUN_DIR)
    whale_count = count_mat_files(WHALE_DIR)
    
    print(f"\nAirgun dataset: {airgun_count} .mat files")
    print(f"  Location: {AIRGUN_DIR}")
    
    print(f"\nWhale dataset: {whale_count} .mat files")
    print(f"  Location: {WHALE_DIR}")
    
    print(f"\nTotal files available: {airgun_count + whale_count}")
    
    print("\n" + "="*60)
    print("IMPORTANT:")
    print("="*60)
    print("By default, the autoencoder script only loads a SAMPLE of files:")
    print("  - Single directory: 15 files (--n-samples)")
    print("  - Multiple directories: 100 files (--tsne-samples)")
    print("\nTo train on ALL files, use: --load-all")
    print("  Example: python Autooencoder_11092025.py --improved-only \\")
    print("           --data-dirs <airgun_dir> <whale_dir> \\")
    print("           --load-all --epochs 100")
    print("\nWARNING: --load-all loads everything into memory!")
    print(f"  Estimated memory needed: ~{(airgun_count + whale_count) * 0.5:.1f} MB")
    print("="*60)

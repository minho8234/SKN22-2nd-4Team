import pandas as pd
import numpy as np
import os
import sys
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.model_selection import train_test_split

# Add src to sys.path to import local modules if not running from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import from the previous step
try:
    from src.data.preprocess_and_split import load_data, preprocess_data, TARGET_COL, RANDOM_STATE
except ImportError:
    # Fallback if running from src/data directly
    from preprocess_and_split import load_data, preprocess_data, TARGET_COL, RANDOM_STATE

OUTPUT_DIR = r'c:\Workspaces\SKN22-2nd-4Team\data\03_resampled'

def get_sampling_strategies():
    """Returns a dictionary of sampling strategies."""
    return {
        'SMOTE': SMOTE(random_state=RANDOM_STATE),
        'SMOTE_Tomek': SMOTETomek(random_state=RANDOM_STATE),
        'SMOTE_ENN': SMOTEENN(random_state=RANDOM_STATE)
    }

def print_class_distribution(y, name="Dataset"):
    """Prints class counts and ratios."""
    counts = y.value_counts().sort_index()
    total = len(y)
    print(f"\n[{name}] Class Distribution:")
    for label, count in counts.items():
        ratio = count / total
        print(f"  Class {label}: {count} ({ratio:.2%})")

def save_resampled_data(X_res, y_res, method_name):
    """Saves resampled X and y to CSV."""
    # Ensure dir exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save X
    x_path = os.path.join(OUTPUT_DIR, f"X_train_{method_name.lower()}.csv")
    X_res.to_csv(x_path, index=False)
    
    # Save y
    y_path = os.path.join(OUTPUT_DIR, f"y_train_{method_name.lower()}.csv")
    y_res.to_csv(y_path, index=False)
    
    print(f"  Saved to: {x_path} & {y_path}")

def main():
    print("=== Starting Sampling Pipeline ===")
    
    # 1. Load & Preprocess (Reuse logic)
    df = load_data()
    df_processed = preprocess_data(df)
    
    # 2. Split (Identical logic to preprocess_and_split.py)
    # Drop rows with missing targets (unlabeled test set)
    df_clean = df_processed.dropna(subset=[TARGET_COL])
    
    y = df_clean[TARGET_COL]
    X = df_clean.drop(columns=[TARGET_COL, 'dataset_source', 'id'], errors='ignore')
    
    # Determine split size
    total_samples = len(X)
    if total_samples == 5000:
        test_size_val = 750
    else:
        test_size_val = 0.15
        
    print(f"Splitting data (Total: {total_samples}, Test Size: {test_size_val})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size_val, 
        random_state=RANDOM_STATE, 
        stratify=y
    )
    
    print(f"Original X_train shape: {X_train.shape}")
    print_class_distribution(y_train, "Original Train")
    
    # 3. Apply Sampling Strategies
    strategies = get_sampling_strategies()
    
    for name, sampler in strategies.items():
        print(f"\n--- Applying {name} ---")
        try:
            X_res, y_res = sampler.fit_resample(X_train, y_train)
            
            # Print verification
            print_class_distribution(y_res, f"Resampled ({name})")
            
            # Save
            save_resampled_data(X_res, y_res, name)
            
        except Exception as e:
            print(f"ERROR applying {name}: {e}")

    # 4. Save the ORIGINAL test set for consistency
    # We must also save the original train set for baseline comparison
    print("\n--- Saving Original Sets ---")
    save_resampled_data(X_train, y_train, "original")
    
    # Save X_test, y_test (Only need to do this once)
    x_test_path = os.path.join(OUTPUT_DIR, "X_test.csv")
    y_test_path = os.path.join(OUTPUT_DIR, "y_test.csv")
    X_test.to_csv(x_test_path, index=False)
    y_test.to_csv(y_test_path, index=False)
    print(f"  Saved Test Set to: {x_test_path} & {y_test_path}")
    
    print("\n=== Sampling Pipeline Completed ===")

if __name__ == "__main__":
    main()

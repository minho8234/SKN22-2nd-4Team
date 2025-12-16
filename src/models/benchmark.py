import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Configuration
DATA_DIR = r'c:\Workspaces\SKN22-2nd-4Team\data\03_resampled'
OUTPUT_DIR = r'c:\Workspaces\SKN22-2nd-4Team\data\04_results'
RANDOM_STATE = 42

def load_dataset(dataset_name):
    """Loads X_train, y_train for a specific sampling strategy, and the common Test set."""
    # Construct filenames based on convention: X_train_smote.csv
    train_x_path = os.path.join(DATA_DIR, f"X_train_{dataset_name.lower()}.csv")
    train_y_path = os.path.join(DATA_DIR, f"y_train_{dataset_name.lower()}.csv")
    
    test_x_path = os.path.join(DATA_DIR, "X_test.csv")
    test_y_path = os.path.join(DATA_DIR, "y_test.csv")
    
    if not os.path.exists(train_x_path):
        raise FileNotFoundError(f"Dataset {dataset_name} not found at {train_x_path}")
        
    X_train = pd.read_csv(train_x_path)
    y_train = pd.read_csv(train_y_path).values.ravel() # Ensure 1D array
    X_test = pd.read_csv(test_x_path)
    y_test = pd.read_csv(test_y_path).values.ravel()
    
    return X_train, y_train, X_test, y_test

def get_models():
    """Returns a dict of models. SVM, LR, ANN are wrapped in Pipelines with Scaling."""
    
    models = {}
    
    # 1. Tree/Ensemble Models (No Scaling needed)
    models['DT'] = DecisionTreeClassifier(random_state=RANDOM_STATE)
    models['RF'] = RandomForestClassifier(random_state=RANDOM_STATE)
    models['XGBoost'] = XGBClassifier(eval_metric='logloss', random_state=RANDOM_STATE, use_label_encoder=False)
    models['LightGBM'] = LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)
    models['CatBoost'] = CatBoostClassifier(verbose=0, random_state=RANDOM_STATE)
    
    # 2. Distance/Gradient-based Models (Need Scaling)
    # Pipeline: Scaler -> Model
    models['ANN'] = make_pipeline(StandardScaler(), MLPClassifier(random_state=RANDOM_STATE, max_iter=1000))
    models['SVM'] = make_pipeline(StandardScaler(), SVC(probability=True, random_state=RANDOM_STATE))
    models['LR'] = make_pipeline(StandardScaler(), LogisticRegression(random_state=RANDOM_STATE, max_iter=1000))
    
    return models

def evaluate_models_on_dataset(dataset_name):
    """
    Train and evaluate all models on a specific dataset (e.g., 'smote').
    Returns: Results DataFrame, and a list of ROC data dictionaries.
    """
    print(f"\n[{dataset_name.upper()}] Loading data...")
    try:
        X_train, y_train, X_test, y_test = load_dataset(dataset_name)
    except FileNotFoundError as e:
        print(e)
        return None, None

    models = get_models()
    results = []
    roc_data = {} # store per model
    
    print(f"[{dataset_name.upper()}] Training {len(models)} models...")
    
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        p = precision_score(y_test, y_pred, zero_division=0)
        r = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_prob)
        
        results.append({
            'Model': name,
            'Precision': p,
            'Recall': r,
            'F1-score': f1,
            'ROC AUC': roc
        })
        
        # ROC Data for plotting
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, roc)
        
        print(f"  > {name}: F1={f1:.4f}, AUC={roc:.4f}")
        
    df_results = pd.DataFrame(results).set_index('Model')
    return df_results, roc_data

def plot_roc_curves(roc_data, dataset_name):
    """Plots and saves ROC curves for all models in one figure."""
    plt.figure(figsize=(10, 8))
    
    for name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
        
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {dataset_name.upper()} Dataset')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    
    save_path = os.path.join(OUTPUT_DIR, f"roc_curve_{dataset_name.lower()}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"  ROC Plot saved to: {save_path}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 4 Dataset Variants from previous step
    datasets = ['original', 'smote', 'smote_tomek', 'smote_enn']
    
    all_benchmarks_completed = True
    
    for ds_name in datasets:
        print(f"\n{'='*40}")
        print(f"Processing Dataset: {ds_name}")
        print(f"{'='*40}")
        
        results_df, roc_data = evaluate_models_on_dataset(ds_name)
        
        if results_df is not None:
            # Save Table
            csv_path = os.path.join(OUTPUT_DIR, f"benchmark_{ds_name.lower()}.csv")
            results_df.to_csv(csv_path)
            print(f"  Table saved to: {csv_path}")
            
            # Print for logs
            print(results_df)
            
            # Plot ROC
            plot_roc_curves(roc_data, ds_name)
        else:
            all_benchmarks_completed = False

    if all_benchmarks_completed:
        print("\nAll benchmarks completed successfully.")
    else:
        print("\nSome benchmarks failed due to missing files.")

if __name__ == "__main__":
    main()

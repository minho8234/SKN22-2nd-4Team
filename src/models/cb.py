import pandas as pd
import numpy as np
import optuna
import os
import sys
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

# --- 설정 ---
RAW_DATA_DIR = r'c:\Workspaces\SKN22-2nd-4Team\data\01_raw' # 경로 확인!
OUTPUT_DIR = r'c:\Workspaces\SKN22-2nd-4Team\data\05_optimized'
RANDOM_STATE = 42
N_TRIALS = 1 

def load_and_split_data():
    """
    Test 파일에 정답(churn)이 없으므로,
    Train 데이터를 쪼개서 학습용(Train)과 검증용(Val)으로 나눕니다.
    """
    train_path = os.path.join(RAW_DATA_DIR, "train.csv")
    test_path = os.path.join(RAW_DATA_DIR, "test.csv") # 제출용 데이터
    
    # 1. 데이터 로드
    df_train_full = pd.read_csv(train_path)
    df_submission = pd.read_csv(test_path) # 정답 없음
    
    # 2. 전처리 (Yes/No -> 1/0 변환)
    # 데이터 컬럼을 확인해서 object 타입인 'yes'/'no' 컬럼만 변환
    target_cols = ['churn', 'international_plan', 'voice_mail_plan']
    
    for col in target_cols:
        # train 데이터 처리
        if col in df_train_full.columns and df_train_full[col].dtype == 'object':
            df_train_full[col] = (df_train_full[col] == 'yes').astype(int)
        
        # submission 데이터 처리 (churn은 없을 수 있음)
        if col in df_submission.columns and df_submission[col].dtype == 'object':
            df_submission[col] = (df_submission[col] == 'yes').astype(int)
            
    # 3. 학습용 데이터 분리
    X = df_train_full.drop('churn', axis=1)
    y = df_train_full['churn']
    
    # 자체 검증을 위해 8:2로 분할 (이게 중요!)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    
    return X_train, y_train, X_val, y_val, df_submission

def find_optimal_threshold(y_true, y_prob):
    best_threshold = 0.5
    best_f1 = 0.0
    thresholds = np.arange(0.01, 1.00, 0.01)
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = thresh
    return best_threshold, best_f1

class ModelOptimizer:
    def __init__(self, X_train, y_train, cat_features):
        self.X_train = X_train
        self.y_train = y_train
        self.cat_features = cat_features
        
    def objective(self, trial):
        # 학습 데이터 안에서도 또 쪼개서 검증 (CV)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        
        params = {
            'boosting_type': 'Ordered',
            'bootstrap_type': 'Bayesian',
            'iterations': 200, 
            'random_state': RANDOM_STATE,
            'verbose': 0,
            'allow_writing_files': False,
            'eval_metric': 'F1',
            'cat_features': self.cat_features,
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'depth': trial.suggest_int('depth', 4, 10),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        }
        model = CatBoostClassifier(**params)
        
        try:
            scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='f1', error_score='raise')
        except Exception as e:
            return 0.0
            
        return scores.mean()

def run_optimization():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading Data & Splitting (Train/Val)...")
    # submission 데이터는 정답이 없으므로 X_submission 변수에 따로 저장
    X_train, y_train, X_val, y_val, df_submission = load_and_split_data()
    
    # CatBoost용 범주형 변수 감지
    cat_features = X_train.select_dtypes(include=['object']).columns.tolist()
    print(f"Detected Categorical Features: {cat_features}")
    
    print(f"\n{'='*40}")
    print(f"Optimizing CatBoost (Trials: {N_TRIALS})...")
    print(f"{'='*40}")
    
    optimizer = ModelOptimizer(X_train, y_train, cat_features)
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=RANDOM_STATE))
    study.optimize(optimizer.objective, n_trials=N_TRIALS, show_progress_bar=True)
    
    print(f"Best CV F1: {study.best_value:.4f}")
    print("Best Params:", study.best_params)
    
    # --- 최종 검증 (Validation Set) ---
    print(f"\nTraining Final Model & Evaluating on Validation Set...")
    
    final_model = CatBoostClassifier(
        boosting_type='Ordered', bootstrap_type='Bayesian', 
        iterations=1000, 
        random_state=RANDOM_STATE, verbose=0, 
        allow_writing_files=False, 
        cat_features=cat_features,
        **study.best_params
    )
    
    # Train 데이터 전체로 학습
    final_model.fit(X_train, y_train)
    
    # 1. 우리끼리 채점 (Validation Set)
    y_prob_val = final_model.predict_proba(X_val)[:, 1]
    
    # 임계값 튜닝
    best_thresh, best_f1_val = find_optimal_threshold(y_val, y_prob_val)
    y_pred_val = (y_prob_val >= best_thresh).astype(int)
    
    # 리포트 출력
    print("\n--- Validation Report (Internal Check) ---")
    print(f"Best Threshold: {best_thresh:.2f}")
    # [수정] 순서 정확하게 (정답, 예측값)
    print(confusion_matrix(y_val, y_pred_val))
    report = classification_report(y_val, y_pred_val)
    print(report)
    
    # Save Report
    result_text = (
        f"Model: CatBoost\n"
        f"Best Params: {study.best_params}\n"
        f"Best CV F1: {study.best_value}\n"
        f"Optimal Threshold: {best_thresh}\n"
        f"Validation F1 (Optimized): {best_f1_val}\n\n"
        f"Confusion Matrix:\n{confusion_matrix(y_val, y_pred_val)}\n\n"
        f"Classification Report:\n{report}"
    )
    
    save_path = os.path.join(OUTPUT_DIR, "catboost_optimization_report.txt")
    with open(save_path, "w") as f:
        f.write(result_text)
    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    run_optimization()
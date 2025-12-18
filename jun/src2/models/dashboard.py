import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from catboost import CatBoostClassifier
import plotly.express as px

# =============================================================================
# 0. Page Config
# =============================================================================
st.set_page_config(
    page_title="📊 고객 이탈 예측 & 손실 관리 대시보드",
    layout="wide",
    page_icon="📉"
)

# =============================================================================
# 1. train.csv 위치 자동 탐색 (🔥 최종 해결책)
# =============================================================================
def find_train_csv(start_path):
    path = start_path
    while True:
        candidate = os.path.join(path, "data", "01_raw", "train.csv")
        if os.path.exists(candidate):
            return candidate, path
        parent = os.path.dirname(path)
        if parent == path:
            return None, None
        path = parent

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH, PROJECT_ROOT = find_train_csv(CURRENT_DIR)

if DATA_PATH is None:
    st.error("❌ data/01_raw/train.csv 를 찾을 수 없습니다.")
    st.stop()

# =============================================================================
# 2. Model / Feature 경로 (dashboard 기준)
# =============================================================================
MODEL_PATH = os.path.join(CURRENT_DIR, "churn_model.cbm")
FEATURE_PATH = os.path.join(CURRENT_DIR, "features.pkl")

# =============================================================================
# 3. 디버깅 정보 (확인용)
# =============================================================================
with st.expander("🛠️ 경로 디버깅 정보", expanded=False):
    st.write("📂 dashboard 위치:", CURRENT_DIR)
    st.write("🏠 프로젝트 루트:", PROJECT_ROOT)
    st.write("📄 데이터 경로:", DATA_PATH)
    st.write("📄 데이터 존재:", os.path.exists(DATA_PATH))
    st.write("📄 모델 존재:", os.path.exists(MODEL_PATH))
    st.write("📄 feature 존재:", os.path.exists(FEATURE_PATH))

# =============================================================================
# 4. Load Data
# =============================================================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    # 전처리
    df['international_plan'] = (df['international_plan'] == 'yes').astype(int)
    df['voice_mail_plan'] = (df['voice_mail_plan'] == 'yes').astype(int)
    df['churn'] = (df['churn'] == 'yes').astype(int)

    charge_cols = [
        'total_day_charge', 'total_eve_charge',
        'total_night_charge', 'total_intl_charge'
    ]
    df['total_bill'] = df[charge_cols].sum(axis=1)

    return df

# =============================================================================
# 5. Load Model
# =============================================================================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ churn_model.cbm 파일이 없습니다.")
        st.stop()

    if not os.path.exists(FEATURE_PATH):
        st.error("❌ features.pkl 파일이 없습니다.")
        st.stop()

    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)

    with open(FEATURE_PATH, "rb") as f:
        features = pickle.load(f)

    return model, features

df = load_data()
model, FEATURES = load_model()

# =============================================================================
# 6. Prediction & Risk Logic
# =============================================================================
X = df[FEATURES]
df['Probability'] = model.predict_proba(X)[:, 1]
df['risk_value'] = df['Probability'] * df['total_bill']

def risk_level(p):
    if p >= 0.85: return "Critical"
    if p >= 0.70: return "Warning"
    if p >= 0.40: return "Attention"
    return "Safe"

df['Risk Level'] = df['Probability'].apply(risk_level)

bill_top20 = df['total_bill'].quantile(0.8)
intl_top20 = df['total_intl_charge'].quantile(0.8)

def assign_strategy(row):
    if row['Probability'] >= 0.85 and row['total_bill'] >= bill_top20:
        return "🚨 VIP 전담 케어"
    if row['number_customer_service_calls'] >= 3:
        return "📞 불만 전담 관리"
    if row['total_intl_charge'] >= intl_top20 and row['international_plan'] == 0:
        return "🌍 국제전화 요금제 제안"
    if row['Probability'] >= 0.75:
        return "💰 요금 할인"
    return "일반 유지 관리"

df['Strategy'] = df.apply(assign_strategy, axis=1)

# =============================================================================
# 7. Sidebar Navigation
# =============================================================================
st.sidebar.title("🛡️ Churn Management")
page = st.sidebar.radio(
    "메뉴",
    ["1️⃣ 현황 진단", "2️⃣ 예측 기반 리스크", "3️⃣ 시뮬레이션", "4️⃣ Action List"]
)

# =============================================================================
# 8. Page 1 – 현황 진단
# =============================================================================
if page == "1️⃣ 현황 진단":
    st.title("🩺 고객 이탈 현황 (AS-IS)")

    c1, c2, c3 = st.columns(3)
    c1.metric("총 고객 수", f"{len(df):,} 명")
    c2.metric("이탈률", f"{df['churn'].mean()*100:.2f}%")
    c3.metric("월 손실액", f"€{df[df['churn']==1]['total_bill'].sum():,.0f}")

    fig = px.histogram(df, x="Probability", nbins=30,
                       title="이탈 확률 분포")
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# 9. Page 2 – 예측 기반 리스크
# =============================================================================
elif page == "2️⃣ 예측 기반 리스크":
    st.title("🔮 예측 기반 이탈 리스크 관리")

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.pie(df, names="Risk Level", title="Risk Level 분포")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        agg = df.groupby("Strategy")['risk_value'].sum().reset_index()
        fig2 = px.bar(agg, x="Strategy", y="risk_value",
                      title="전략별 기대 손실액 (€)")
        st.plotly_chart(fig2, use_container_width=True)

# =============================================================================
# 10. Page 3 – 시뮬레이션
# =============================================================================
elif page == "3️⃣ 시뮬레이션":
    st.title("🎛️ 이탈 방지 시뮬레이션")

    improve = st.slider("이탈 방어 성공률 (%)", 0, 100, 20, step=5)
    saved = df['risk_value'].sum() * (improve / 100)

    st.metric("💰 방어 가능한 예상 매출", f"€{saved:,.0f}")

# =============================================================================
# 11. Page 4 – Action List
# =============================================================================
elif page == "4️⃣ Action List":
    st.title("📋 실전 고객 관리 리스트")

    target_df = df[df['Risk Level'].isin(["Critical", "Warning"])] \
        .sort_values("risk_value", ascending=False)

    display_df = target_df[
        ['Risk Level', 'Probability', 'total_bill', 'risk_value', 'Strategy']
    ].copy()

    display_df['Probability'] *= 100

    st.dataframe(
        display_df,
        column_config={
            "Probability": st.column_config.ProgressColumn(
                "이탈 확률 (%)", min_value=0, max_value=100
            )
        },
        use_container_width=True,
        height=500
    )

    st.download_button(
        "📥 CSV 다운로드",
        display_df.to_csv(index=False).encode("utf-8-sig"),
        "churn_action_list.csv",
        "text/csv"
    )

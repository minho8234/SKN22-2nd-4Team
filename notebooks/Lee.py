import streamlit as st
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import Pool, CatBoostClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('../data/01_raw/train.csv')

df['churn'] = (df['churn'] == 'yes').astype(int)
df['international_plan'] = (df['international_plan'] == 'yes').astype(int)
df['voice_mail_plan'] = (df['voice_mail_plan'] == 'yes').astype(int)

df['international_plan_label'] = df['international_plan'].map({0: 'No', 1: 'Yes'})
df['voice_mail_plan_label'] = df['voice_mail_plan'].map({0: 'No', 1: 'Yes'})

X = df.drop('churn', axis=1)
y = df['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y)

cat_features = X.select_dtypes(include='object').columns.tolist()

# train_pool = Pool(X, y, cat_features=cat_features)
train_pool = Pool(X_train, y_train, cat_features=cat_features)

model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    eval_metric='AUC',
    verbose=False
)

model.fit(train_pool)
st.title("Churn Rate Explorer")




target_col = 'churn'

filter_cols = [
    'international_plan_label',
    'voice_mail_plan_label'
]

feature_cols = [c for c in df.columns if c not in [target_col]]

cat_cols = [
    'international_plan_label',
    'voice_mail_plan_label'
]

num_cols = [
    c for c in feature_cols
    if c not in cat_cols and df[c].dtype != 'object'
]

st.sidebar.header("조건 선택")

filters = {}

for col in cat_cols:
    options = df[col].dropna().unique().tolist()
    selected = st.sidebar.multiselect(
        f"{col} 선택",
        options,
        default=options
    )
    filters[col] = selected

for col in num_cols:
    min_val = float(df[col].min())
    max_val = float(df[col].max())

    selected_range = st.sidebar.slider(
        f"{col} 범위",
        min_val, max_val,
        (min_val, max_val)
    )
    filters[col] = selected_range

filtered_df = df.copy()

for col, condition in filters.items():
    if col in cat_cols:
        if col == 'international_plan_label':
            filtered_df = filtered_df[
                filtered_df['international_plan'].isin(
                    [1 if v == 'Yes' else 0 for v in condition]
                )
            ]
        elif col == 'voice_mail_plan_label':
            filtered_df = filtered_df[
                filtered_df['voice_mail_plan'].isin(
                    [1 if v == 'Yes' else 0 for v in condition]
                )
            ]
    else:
        filtered_df = filtered_df[
            (filtered_df[col] >= condition[0]) &
            (filtered_df[col] <= condition[1])
        ]

st.subheader("이탈 분석 결과")
if len(filtered_df) > 0:
    churn_rate = filtered_df['churn'].mean()
else:
    churn_rate = 0.0
st.metric(
    label="이탈률 (Churn Rate)",
    value=f"{churn_rate * 100:.2f} %"
)

st.write(f"대상 고객 수: {len(filtered_df)} 명")

if filtered_df.empty:
    st.warning("선택한 조건에 해당하는 데이터가 없습니다.")



counts = filtered_df['churn'].value_counts()

fig, ax = plt.subplots()
counts.plot(kind='bar', ax=ax)
ax.set_xticklabels(['No Churn', 'Churn'], rotation=0)
ax.set_ylabel("Count")

st.pyplot(fig)

if st.sidebar.button("조건 초기화"):
    st.experimental_rerun()

if filtered_df.empty:
    st.warning("선택한 조건에 해당하는 데이터가 없습니다.")

# Feature Importance (모델 해석)
importances = model.get_feature_importance()
feature_names = model.feature_names_

imp_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

st.subheader("이탈에 영향을 많이 준 변수")
st.bar_chart(imp_df.set_index('feature').head(10))

# 개별 고객 이탈 확률 예측 (실무 핵심)
st.divider()
st.header("📌 개별 고객 이탈 확률 예측")

st.write("고객 정보를 입력하면 이탈 확률을 예측합니다.")

# =========================
# 1. 고객 정보 입력 UI
# =========================

st.subheader("🧾 고객 정보 입력")

# --- Categorical ---
state = st.selectbox(
    "State",
    options=sorted(df['state'].unique())
)

area_code = st.selectbox(
    "Area Code",
    options=sorted(df['area_code'].unique())
)

# --- Binary ---
international_plan = st.selectbox(
    "International Plan",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

voice_mail_plan = st.selectbox(
    "Voice Mail Plan",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

# --- Numeric ---
account_length = st.slider(
    "Account Length",
    int(df['account_length'].min()),
    int(df['account_length'].max()),
    int(df['account_length'].median())
)

number_vmail_messages = st.slider(
    "Number of Voice Mail Messages",
    int(df['number_vmail_messages'].min()),
    int(df['number_vmail_messages'].max()),
    int(df['number_vmail_messages'].median())
)

total_day_minutes = st.slider(
    "Total Day Minutes",
    float(df['total_day_minutes'].min()),
    float(df['total_day_minutes'].max()),
    float(df['total_day_minutes'].median())
)

total_day_calls = st.slider(
    "Total Day Calls",
    int(df['total_day_calls'].min()),
    int(df['total_day_calls'].max()),
    int(df['total_day_calls'].median())
)

total_day_charge = st.slider(
    "Total Day Charge",
    float(df['total_day_charge'].min()),
    float(df['total_day_charge'].max()),
    float(df['total_day_charge'].median())
)

total_eve_minutes = st.slider(
    "Total Evening Minutes",
    float(df['total_eve_minutes'].min()),
    float(df['total_eve_minutes'].max()),
    float(df['total_eve_minutes'].median())
)

total_eve_calls = st.slider(
    "Total Evening Calls",
    int(df['total_eve_calls'].min()),
    int(df['total_eve_calls'].max()),
    int(df['total_eve_calls'].median())
)

total_eve_charge = st.slider(
    "Total Evening Charge",
    float(df['total_eve_charge'].min()),
    float(df['total_eve_charge'].max()),
    float(df['total_eve_charge'].median())
)

total_night_minutes = st.slider(
    "Total Night Minutes",
    float(df['total_night_minutes'].min()),
    float(df['total_night_minutes'].max()),
    float(df['total_night_minutes'].median())
)

total_night_calls = st.slider(
    "Total Night Calls",
    int(df['total_night_calls'].min()),
    int(df['total_night_calls'].max()),
    int(df['total_night_calls'].median())
)

total_night_charge = st.slider(
    "Total Night Charge",
    float(df['total_night_charge'].min()),
    float(df['total_night_charge'].max()),
    float(df['total_night_charge'].median())
)

total_intl_minutes = st.slider(
    "Total International Minutes",
    float(df['total_intl_minutes'].min()),
    float(df['total_intl_minutes'].max()),
    float(df['total_intl_minutes'].median())
)

total_intl_calls = st.slider(
    "Total International Calls",
    int(df['total_intl_calls'].min()),
    int(df['total_intl_calls'].max()),
    int(df['total_intl_calls'].median())
)

total_intl_charge = st.slider(
    "Total International Charge",
    float(df['total_intl_charge'].min()),
    float(df['total_intl_charge'].max()),
    float(df['total_intl_charge'].median())
)

number_customer_service_calls = st.slider(
    "Customer Service Calls",
    int(df['number_customer_service_calls'].min()),
    int(df['number_customer_service_calls'].max()),
    int(df['number_customer_service_calls'].median())
)

# =========================
# 2. customer_df 생성
# ⚠ 반드시 학습에 사용한 컬럼만
# =========================

customer_df = pd.DataFrame([{
    # Categorical
    'state': state,                      # 예: 'CA'
    'area_code': area_code,              # 예: 415

    # Binary (0 / 1)
    'international_plan': international_plan,  # 0 or 1
    'voice_mail_plan': voice_mail_plan,        # 0 or 1

    # Numeric
    'account_length': account_length,
    'number_vmail_messages': number_vmail_messages,

    'total_day_minutes': total_day_minutes,
    'total_day_calls': total_day_calls,
    'total_day_charge': total_day_charge,

    'total_eve_minutes': total_eve_minutes,
    'total_eve_calls': total_eve_calls,
    'total_eve_charge': total_eve_charge,

    'total_night_minutes': total_night_minutes,
    'total_night_calls': total_night_calls,
    'total_night_charge': total_night_charge,

    'total_intl_minutes': total_intl_minutes,
    'total_intl_calls': total_intl_calls,
    'total_intl_charge': total_intl_charge,

    'number_customer_service_calls': number_customer_service_calls
}])
X_columns = X_train.columns.tolist()
customer_df = customer_df[X_train.columns]

st.subheader("입력된 고객 정보")
st.dataframe(customer_df)

# =========================
# 3. 이탈 확률 예측
# =========================

if st.button("이탈 확률 예측"):
    proba = model.predict_proba(customer_df)[0, 1]

    st.metric(
        label="📊 예측 이탈 확률",
        value=f"{proba * 100:.2f} %",
        delta="High Risk 🚨" if proba >= 0.7 else "Low Risk ✅"
    )

    if proba >= 0.7:
        st.error("⚠ 이 고객은 이탈 가능성이 높습니다. 적극적인 관리가 필요합니다.")
    else:
        st.success("✅ 이 고객은 이탈 가능성이 낮습니다.")



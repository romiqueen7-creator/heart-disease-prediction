import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

# ==========================================
# 1. 페이지 설정 및 제목
# ==========================================
st.set_page_config(page_title="AI 심장 진단 솔루션", page_icon="❤️")

st.title("🏥 AI 심장 질환 예측 솔루션 v3.0")
st.markdown("""
이 프로그램은 **딥러닝(Deep Learning)** 기술을 활용하여 
사용자의 건강 데이터를 분석하고 심장 질환 위험도를 예측합니다.
*(데이터 출처: UCI Machine Learning Repository - 4개국 통합 데이터)*
""")

# ==========================================
# 2. 데이터 로드 및 모델 학습 (캐싱 적용)
# ==========================================
# @st.cache_resource를 쓰면 새로고침할 때마다 학습하지 않고 한 번만 학습합니다.
@st.cache_resource
def train_model():
    with st.spinner('AI가 전 세계 4개국 병원 데이터를 학습 중입니다... 잠시만 기다려주세요.'):
        # 데이터 로드
        urls = [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data"
        ]
        column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
        
        dfs = [pd.read_csv(url, names=column_names, na_values="?", on_bad_lines='skip') for url in urls]
        df = pd.concat(dfs, ignore_index=True)

        # 전처리
        features = ['age', 'sex', 'chol', 'thalach', 'trestbps', 'cp', 'oldpeak']
        df_clean = df[features + ['target']].dropna()
        y = np.where(df_clean['target'] > 0, 1, 0).reshape(-1, 1)
        X = df_clean[features].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 딥러닝 클래스 정의 (내부 함수용)
        class DeepHeartModel:
            def __init__(self, input_size, hidden_size, output_size):
                self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
                self.b1 = np.zeros((1, hidden_size))
                self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
                self.b2 = np.zeros((1, output_size))

            def sigmoid(self, x): return 1 / (1 + np.exp(-x))
            def relu(self, x): return np.maximum(0, x)
            def relu_deriv(self, x): return (x > 0).astype(float)

            def forward(self, X):
                self.z1 = np.dot(X, self.W1) + self.b1
                self.a1 = self.relu(self.z1)
                self.z2 = np.dot(self.a1, self.W2) + self.b2
                self.a2 = self.sigmoid(self.z2)
                return self.a2

            def train(self, X, y, epochs, learning_rate):
                m = y.shape[0]
                for _ in range(epochs):
                    output = self.forward(X)
                    error = output - y
                    dW2 = (1 / m) * np.dot(self.a1.T, error)
                    db2 = (1 / m) * np.sum(error, axis=0, keepdims=True)
                    error_hidden = np.dot(error, self.W2.T) * self.relu_deriv(self.z1)
                    dW1 = (1 / m) * np.dot(X.T, error_hidden)
                    db1 = (1 / m) * np.sum(error_hidden, axis=0, keepdims=True)
                    self.W1 -= learning_rate * dW1
                    self.b1 -= learning_rate * db1
                    self.W2 -= learning_rate * dW2
                    self.b2 -= learning_rate * db2
        
        # 모델 학습
        model = DeepHeartModel(input_size=7, hidden_size=24, output_size=1)
        model.train(X_scaled, y, epochs=5000, learning_rate=0.01)
        
        return model, scaler

# 모델 로드 (최초 1회만 실행됨)
try:
    model, scaler = train_model()
    st.success("✅ AI 모델 학습 및 준비 완료!")
except Exception as e:
    st.error(f"데이터 로드 중 오류 발생: {e}")
    st.stop()

# ==========================================
# 3. 사용자 입력 (사이드바)
# ==========================================
st.sidebar.header("📝 환자 정보 입력")

age = st.sidebar.slider("나이 (Age)", 10, 100, 45)
sex_option = st.sidebar.radio("성별 (Sex)", ("남성", "여성"))
sex = 1 if sex_option == "남성" else 0

st.sidebar.markdown("---")
cp_option = st.sidebar.selectbox(
    "흉통 유형 (Chest Pain Type)", 
    ("1: 전형적 협심증", "2: 비전형적 협심증", "3: 통증 없음", "4: 무증상")
)
cp = int(cp_option[0]) # 앞의 숫자만 가져옴

trestbps = st.sidebar.number_input("안정 시 혈압 (mm Hg)", 80, 200, 120)
chol = st.sidebar.number_input("콜레스테롤 (mg/dl)", 100, 600, 200)
thalach = st.sidebar.slider("최대 심박수 (Max Heart Rate)", 60, 220, 150)
oldpeak = st.sidebar.slider("ST 우울 수치 (Oldpeak)", 0.0, 6.0, 0.0, step=0.1)

# ==========================================
# 4. 예측 및 결과 시각화
# ==========================================
if st.sidebar.button("🔍 진단 결과 확인하기", type="primary"):
    # 입력 데이터 전처리
    input_data = np.array([[age, sex, chol, thalach, trestbps, cp, oldpeak]])
    input_scaled = scaler.transform(input_data)
    
    # 예측
    prediction = model.forward(input_scaled)[0][0]
    percent = prediction * 100
    
    st.divider()
    st.subheader("📊 AI 진단 리포트")
    
    # 게이지 바 시각화
    st.progress(int(percent))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="심장 질환 발병 확률", value=f"{percent:.2f}%")
    
    with col2:
        if percent >= 50:
            st.error("판정: 고위험군 (High Risk)")
            st.write("🔴 **경고:** 전문의와의 정밀 상담이 강력히 권장됩니다.")
        else:
            st.success("판정: 저위험군 (Low Risk)")
            st.write("🟢 **정상:** 현재 심장 건강 상태는 양호합니다.")
            
    # 추가 조언
    st.info(f"""
    **[참고 분석]**
    - 입력하신 **{age}세 {sex_option}**의 데이터 기준입니다.
    - 콜레스테롤({chol})과 혈압({trestbps}) 관리에 유의하세요.
    """)

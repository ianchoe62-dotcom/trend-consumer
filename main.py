import os
import re
from collections import Counter
import joblib
import matplotlib.pyplot as plt
import pandas as pd  # <-- 이 부분이 상단에 반드시 있어야 합니다!
import plotly.express as px
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import streamlit as st
import wordcloud
import os
import re
from collections import Counter
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import streamlit as st
import wordcloud

# --- 1, 2, 4단계: 데이터 자동 생성, 전처리 및 모델 학습 (최초 실행 시 자동 처리) ---
@st.cache_resource
def initialize_data_and_model():
    if not os.path.exists('raw_data.csv'):
        data = {
            'date': ['2026-03-01', '2026-03-02', '2026-03-03'],
            'source': ['News', 'Twitter', 'Blog'],
            'title': ['제로슈거 음료 인기', '제로 음료 맛 후기', '친환경 트렌드 분석'],
            'content': [
                '제로슈거 음료 맛도 깔끔하고 살 안 찐다니까 매일 마시게 됨. 너무 좋다!',
                '제로 음료에 들어가는 대체당 성분 장기 복용하면 부작용 있다는 연구 결과 나왔는데 무분별하게 마시네...',
                '인스타에서 유행하길래 궁금해서 한 번 사봤음. 인증샷 남기기 딱 좋네.'
            ],
            'label': ['긍정적 수용', '부정적 반발', '단순 소비']
        }
        df = pd.DataFrame(data)
        df.to_csv('raw_data.csv', index=False, encoding='utf-8-sig')

    if not os.path.exists('preprocessed_data.csv'):
        df = pd.read_csv('raw_data.csv')
        stopwords = ['은', '는', '이', '가', '을', '를', '의', '에', '도', '고', '으로', '로', '하', '는', '것']
        
        clean_texts = []
        for text in df['content'].fillna(''):
            text = re.sub(r'<.*?>', '', text)
            text = re.sub(r'http\S+', '', text)
            text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text)
            words = text.split()
            tokens = [word for word in words if word not in stopwords and len(word) > 1]
            clean_texts.append(' '.join(tokens))
            
        df['clean_text'] = clean_texts
        df.to_csv('preprocessed_data.csv', index=False, encoding='utf-8-sig')

    if not os.path.exists('sentiment_model.pkl') or not os.path.exists('tfidf_vectorizer.pkl'):
        df = pd.read_csv('preprocessed_data.csv')
        X = df['clean_text'].fillna('')
        y = df['label']
        
        vectorizer = TfidfVectorizer()
        X_vec = vectorizer.fit_transform(X)
        
        model = LogisticRegression()
        model.fit(X_vec, y)
        
        joblib.dump(model, 'sentiment_model.pkl')
        joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

    df_loaded = pd.read_csv('preprocessed_data.csv')
    model_loaded = joblib.load('sentiment_model.pkl')
    vectorizer_loaded = joblib.load('tfidf_vectorizer.pkl')
    return model_loaded, vectorizer_loaded, df_loaded

model, vectorizer, df = initialize_data_and_model()

# --- 5단계: Streamlit 대시보드 웹 앱 UI ---
st.title("📊 SNS 및 뉴스 트렌드 대중 반응 분석 대시보드")

# 사이드바: 전체 반응 비율 피자 차트
st.sidebar.header("전체 데이터 반응 비율")
if 'label' in df.columns:
    label_counts = df['label'].value_counts().reset_index()
    label_counts.columns = ['Label', 'Count']
    fig_pie = px.pie(label_counts, names='Label', values='Count', title="반응 유형 분포")
    st.sidebar.plotly_chart(fig_pie)

# 메인 화면: 텍스트 입력 및 예측
st.subheader("새로운 텍스트 반응 분류 예측")
user_input = st.text_area("뉴스 기사 또는 SNS 텍스트를 입력하세요:")

if st.button("분류 예측 실행"):
    if user_input.strip() == "":
        st.warning("텍스트를 입력해주세요.")
    else:
        vec = vectorizer.transform([user_input])
        pred = model.predict(vec)[0]
        probs = model.predict_proba(vec)[0]
        classes = model.classes_
        
        st.success(f"**예측된 대중 반응 유형:** `{pred}`")
        
        # 확률 분포 바 차트 시각화
        prob_df = pd.DataFrame({'Reaction': classes, 'Probability': probs})
        fig_bar = px.bar(prob_df, x='Reaction', y='Probability', color='Reaction', title="유형별 확률 분포")
        st.plotly_chart(fig_bar)
import pandas as pd
scikit-learn
matplotlib
seaborn
wordcloud
streamlit
plotly
joblib

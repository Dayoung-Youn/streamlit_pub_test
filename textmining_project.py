import streamlit as st
import pandas as pd
import plotly.express as px
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from ast import literal_eval


# 사이드바에 페이지 선택 옵션 설정
page = st.sidebar("페이지 선택", ["textmining project", "dictionary", "mpb tone and BOK baserate"])

# 각 페이지에 대한 내용 설정
if page == "textmining project":
    st.header("textmining project")
    # 페이지 1 내용 설정
elif page == "dictionary":
    st.header("dictionary")
    # 페이지 2 내용 설정
elif page == "mpb tone and BOK baserate":
    st.header("mpb tone and BOK baserate")
    # 페이지 3 내용 설정

st.title("TEXTMINING PROJECT")

st.header("Summary")

#summary='**금통위 의사록을 분석해 한국은행의 기준금리 향방을 예측하는 논문을 구현하는 프로젝트를 진행했습니다.**\n\n \t먼저, 뉴스기사, 채권보고서, mpb회의록을 크롤링하여 날짜를 기준으로 데이터프레임을 만들어 정리하고, 각 날짜에 해당하는 기준금리를 한달 전 대비 상승 또는 하강했는지 라벨링하였습니다.\n \t다음으로 한국어를 위한 경제금융 사전인 eKoPLNY를 활용해 문서별 ngram을 구하고, 극성점수가 1.3이상인 ngram들을 모아 hawkish dictionary 리스트를, 극성점수가 0.76 이하인 ngram들을 모아 dovish dictionary 리스트를 만들어 사전을 만들었습니다.\n \t이후 이 사전과 2014년부터 2024년 동안의 금통위 의사록의 ngram을 비교해 문서의 어조를 구하고 기준금리 향방을 예측했습니다.\n\n 이하 페이지는 이 과정을 streamlit으로 시각화한 내용입니다.'
#st.markdown(summary)
#st.text_area(label='', value=summary, height=230)

summary = (
    "**금통위 의사록을 분석해 한국은행의 기준금리 향방을 예측하는 논문을 구현하는 프로젝트를 진행했습니다.**<br><br>"
    "&emsp;먼저, 뉴스기사, 채권보고서, mpb회의록을 크롤링하여 날짜를 기준으로 데이터프레임을 만들어 정리하고, "
    "각 날짜에 해당하는 기준금리를 한달 전 대비 상승 또는 하강했는지 라벨링하였습니다.<br>"
    "&emsp;다음으로 한국어를 위한 경제금융 사전인 eKoNLPy를 활용해 문서별 ngram을 구하고, 극성점수가 1.3이상인 ngram들을 "
    "모아 hawkish dictionary 리스트를, 극성점수가 0.76 이하인 ngram들을 모아 dovish dictionary 리스트를 만들어 사전을 만들었습니다.<br>"
    "&emsp;이후 이 사전과 2014년부터 2024년 동안의 금통위 의사록의 ngram을 비교해 문서의 어조를 구하고 기준금리 향방을 예측했습니다.<br><br>"
    "이하 페이지는 이 과정을 streamlit으로 시각화한 내용입니다."
)

st.markdown(summary, unsafe_allow_html=True)

st.header("Proportion of Corpus")
corpus=pd.read_csv('corpus.csv')

def main():
    #st.title("자료별 비중을 나타내는 원형 차트")

    # Plotly를 사용하여 원형 차트 생성
    fig = px.pie(corpus, values='파일의 수', names='파일 유형')
    colors = {
        'news': 'lightgreen',  # 연두색
        'mpb': 'yellow',       # 노란색
        'financial report': 'lightblue'}

    # 각 자료명에 대해 색상 설정
    fig.update_traces(marker=dict(colors=[colors[name] for name in corpus['파일 유형']]))


    # Streamlit을 사용하여 차트 표시
    st.plotly_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()

#단어사전 개요
st.header("Word Dictionary")

col1, col2 = st.columns(2)
col2.metric(label="hawkish dictionary", value="1950개")
col1.metric(label="dovish dictionary", value="3031개")

hdict=pd.read_csv("hawkish_re_list.csv")
df1=pd.DataFrame(hdict)
df1=df1.drop(columns=['Unnamed: 0', 'count_down'])
df1 =df1.rename(columns={'count_up': 'count_hawkish'})
df1.to_string(index=False)

col2.dataframe(df1)

ddict=pd.read_csv("dovish_re_list.csv")
ddict=pd.DataFrame(ddict)
ddict=ddict.drop(columns=['Unnamed: 0', 'count_up'])
ddict = ddict.rename(columns={'count_down': 'count_dovish'})

col1.dataframe(ddict)

# CSV 파일 로드
df_dict = pd.read_csv('dict.csv')
df_dict = df_dict.rename(columns={'count_up': 'count_hawkish', 'count_down': 'count_dovish'})
df_dict=df_dict.drop(columns=['polarity_score'])
dict=pd.read_csv('dict.csv')


# 사용자 입력을 받는 섹션
st.header("Search for Ngram Polarity Scores")

st.markdown("<i>ngram별 pos tagging을 알아야 극성점수 검색이 가능하므로 아래 사전을 참고하세요.</i>", unsafe_allow_html=True)

#st.caption("ngram별 pos tagging을 알아야 극성점수 검색이 가능하므로 아래 사전을 참고하세요.")
st.dataframe(df_dict)

search_word = st.text_input("단어를 입력하세요")

if search_word:
    # 입력된 단어를 소문자로 변환 (일치성 확인을 위해)
    search_word = search_word.lower()

    # 단어 검색
    result = dict[dict['words'].str.lower() == search_word]

    if not result.empty:
        polarity_score = result['polarity_score'].values[0]
        st.write(f"'{search_word}'의 극성 점수는 {polarity_score} 입니다.")
    else:
        st.write(f"'{search_word}'는 사전에 없습니다.")




#산점도 그래프
df2 = pd.read_csv('merge_df (1).csv')  

# 문자열을 float으로 변환하는 함수
def convert_to_float(x):
    try:
        value = literal_eval(x)
        if isinstance(value, list):
            return float(value[0])
        return float(value)
    except (ValueError, SyntaxError):
        return None

df2['tone_doc'] = df2['tone_doc'].apply(convert_to_float)
#df2['baserate'] = pd.to_numeric(df2['baserate'], errors='coerce')

# NaN 값 제거 또는 처리 (예: NaN 값이 있는 행 제거)
df2 = df2.dropna(subset=['tone_doc', 'baserate'])


# Min-Max 정규화
# scaler = MinMaxScaler()
# df2[['tone_doc', 'baserate']] = scaler.fit_transform(df2[['tone_doc', 'baserate']])


if df2.empty:
    st.error("DataFrame is empty after dropping NaN values. Please check your data.")
else:
    # Min-Max 정규화
    scaler = MinMaxScaler()
    df2[['tone_doc', 'baserate']] = scaler.fit_transform(df2[['tone_doc', 'baserate']])

    # 날짜 흐름에 따른 doc_tone과 base_rate의 상관관계 산점도
    df2['date'] = pd.to_datetime(df2['date'], format='%Y-%m-%d').dt.strftime('%Y-%m-%d')

#df2['date'] = pd.to_datetime(df2['date'], format='%Y-%m-%d')
# 날짜 흐름에 따른 doc_tone과 base_rate의 상관관계 산점도
    st.header("The Correlation Scatter Plot between MPB minutes tone and BOK baserate")
    # st.scatter_chart를 사용하여 산점도 그리기
        #st.title("Scatter Plot with Trendline")

        # Assuming merge_df is already defined and contains 'tone_doc', 'baserate', and 'date' columns
    fig = px.scatter(df2, x='tone_doc', y='baserate', color='date',
                        labels={'tone_doc': 'MPB Tone', 'baserate': 'BOK baserate'},
                        trendline='ols', title='Correlation between MPB minutes tone and BOK baserate')
        # Streamlit에서 Plotly 차트 표시
    st.plotly_chart(fig, use_container_width=True)

        # 색상 진하게 설정
    #fig.update_traces(marker=dict(color='navy', size=10, opacity=0.8))

    #st.plotly_chart(fig)

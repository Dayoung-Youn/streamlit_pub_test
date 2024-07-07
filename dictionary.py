import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Dictionary Filter")

col1, col2 = st.columns(2)
 # Streamlit 웹 페이지 제목
col2.header("Hawkish Dictionary")
hdict=pd.read_csv("hawkish_list.csv")
df1=pd.DataFrame(hdict)
df1=df1.drop(columns=['count_down'])
df1 =df1.rename(columns={'count_up': 'count_hawkish'})
df1.to_string(index=False)
#col2.dataframe(df1)

col1.header("Dovish Dictionary")
ddict=pd.read_csv("/Users/trini.y/Downloads/dovish_list.csv")
ddict=pd.DataFrame(ddict)
ddict=ddict.drop(columns=['count_up'])
ddict = ddict.rename(columns={'count_down': 'count_dovish'})

#col1.dataframe(ddict)

    # polarity score의 최소값과 최대값 확인
min_score = df1['polarity_score'].min()
max_score = df1['polarity_score'].max()

    # 슬라이더 설정
score_range = col2.slider("확인하고 싶은 극성점수의 범주를 설정하세요.", min_value=min_score, max_value=max_score, value=(min_score, max_score), step=0.1)

    # 선택된 범위에 해당하는 단어 필터링
filtered_df = df1[(df1['polarity_score'] >= score_range[0]) & (df1['polarity_score'] <= score_range[1])]

    # 결과 출력
if not filtered_df.empty:
    col2.subheader(f"극성점수가 {score_range[0]}과 {score_range[1]} 사이에 있는 단어는 다음과 같습니다.")
    col2.write(filtered_df)
else:
    col2.subheader("선택된 범주 내의 극성점수에 해당하는 단어가 없습니다.")






#col1.header("Dovish Dictionary Filter")

    # polarity score의 최소값과 최대값 확인
min_score = ddict['polarity_score'].min()
max_score = ddict['polarity_score'].max()

    # 슬라이더 설정
score_range = col1.slider("확인하고 싶은 극성점수의 범주를 설정하세요.", min_value=min_score, max_value=max_score, value=(min_score, max_score), step=0.1)

    # 선택된 범위에 해당하는 단어 필터링
filtered_df = ddict[(ddict['polarity_score'] >= score_range[0]) & (ddict['polarity_score'] <= score_range[1])]

    # 결과 출력
if not filtered_df.empty:
    col1.subheader(f"극성점수가 {score_range[0]}과 {score_range[1]} 사이에 있는 단어는 다음과 같습니다.")
    col1.write(filtered_df)
else:
    col1.subheader("선택된 범주 내의 극성점수에 해당하는 단어가 없습니다.")




# count_hawkish와 count_dovish 값을 바탕으로 상위 10개 단어 추출
top_10_hawkish = df1.nlargest(10, 'count_hawkish')
top_10_dovish = ddict.nlargest(10, 'count_dovish')

# Streamlit 앱 구성
st.header("사전별 빈도수 상위 10개 ngram")

col1, col2 = st.columns(2)

col2.subheader("Hawkish ngram")
fig_hawkish = px.bar(top_10_hawkish, x='words', y='count_hawkish', title='Top 10 Hawkish Words', labels={'count_hawkish': 'Hawkish Count', 'words': 'Words'})
fig_hawkish.update_traces(marker_color='red')
col2.plotly_chart(fig_hawkish)

col1.subheader("Dovish ngram")
fig_dovish = px.bar(top_10_dovish, x='words', y='count_dovish', title='Top 10 Dovish Words', labels={'count_dovish': 'Dovish Count', 'words': 'Words'})
col1.plotly_chart(fig_dovish)


import streamlit as st
import pandas as pd
import plotly.express as px
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from ast import literal_eval
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import json
import requests
import datetime

# CSV 파일 로드
df = pd.read_csv('dict.csv')
df2 = pd.read_csv('merge_df (1).csv')  

hdict=pd.read_csv("hawkish_list.csv")
hdict=pd.DataFrame(hdict)

ddict=pd.read_csv("dovish_list.csv")
ddict=pd.DataFrame(ddict)

#df2['tone_doc'] = pd.to_numeric(df2['tone_doc'], errors='coerce')
#df2['baserate'] = pd.to_numeric(df2['baserate'], errors='coerce')

#df2['tone_doc'].fillna(df2['tone_doc'].median(), inplace=True)
#df2['baserate'].fillna(df2['baserate'].median(), inplace=True)

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
scaler = MinMaxScaler()
df2[['tone_doc', 'baserate']] = scaler.fit_transform(df2[['tone_doc', 'baserate']])


if df2.empty:
    st.error("DataFrame is empty after dropping NaN values. Please check your data.")
else:
    # Min-Max 정규화
    #scaler = MinMaxScaler()
    #df2[['tone_doc', 'baserate']] = scaler.fit_transform(df2[['tone_doc', 'baserate']])


    # 날짜 흐름에 따른 doc_tone과 base_rate의 선 그래프
    st.header('금통위의사록 어조와 기준금리 변화')
    # 데이터를 시간 순서로 정렬
    df2 = df2.sort_values(by='date')
   # 인덱스로 설정하기 전에 원래 상태 유지
    df2.reset_index(drop=True, inplace=True)

    # st.line_chart를 사용하여 선 그래프 그리기
    st.line_chart(df2[['date', 'tone_doc', 'baserate']].set_index('date'), use_container_width=True)

    # 데이터 프레임 출력 (디버깅용)
    #st.write(df2[['date', 'tone_doc', 'baserate']])

    #기간 선택
    # Convert date format
    df2['date'] = pd.to_datetime(df2['date'], format='%Y-%m-%d')

    # Sort data by date
    df2 = df2.sort_values(by='date')

    # Reset index
    df2.reset_index(drop=True, inplace=True)

    # Streamlit app title
    #st.title('금통위의사록 어조와 기준금리 변화')

    col1, col2=st.columns(2)
    # Date range selection
    min_date = datetime.date(2014, 3, 13)
    max_date = datetime.date(2024, 5, 23)
    default_date = datetime.date(2024, 5, 23)
    if default_date < min_date or default_date > max_date:
        default_date = min_date 
    
    start_date = col1.date_input("시작 날짜", min_value=pd.Timestamp(df2['date'].min()).date(), max_value=pd.Timestamp(df2['date'].max()).date())
    end_date = col2.date_input("종료 날짜", min_value=pd.Timestamp(df2['date'].min()).date(), max_value=pd.Timestamp(df2['date'].max()).date())

    # Filter data based on selected date range
    mask = (df2['date'] >= pd.Timestamp(start_date)) & (df2['date'] <= pd.Timestamp(end_date))
    filtered_df = df2.loc[mask]

    # Line chart with tone_doc and baserate over time
    fig = px.line(filtered_df, x='date', y=['tone_doc', 'baserate'],
                  labels={'value': '값', 'date': '날짜', 'variable': '변수'}, title='기간별 금통위의사록 어조와 기준금리 변화')

    # Plotly chart with Streamlit
    st.plotly_chart(fig, use_container_width=True)


merge_df=pd.read_csv('minutes_new_count.csv')
merge_df=merge_df.drop(columns=['Unnamed: 0', 'split_content', 'tone_sentence'])
merge_df['tone_doc'] = merge_df['tone_doc'].apply(convert_to_float)

class CompletionExecutor:
    def __init__(self, host, api_key, api_key_primary_val, request_id):
        self._host = host
        self._api_key = api_key
        self._api_key_primary_val = api_key_primary_val
        self._request_id = request_id

    def execute(self, completion_request):
        headers = {
            'X-NCP-CLOVASTUDIO-API-KEY': self._api_key,
            'X-NCP-APIGW-API-KEY': self._api_key_primary_val,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id,
            'Content-Type': 'application/json; charset=utf-8',
            'Accept': 'text/event-stream'
        }
        ret = []
        with requests.post(self._host + '/testapp/v1/chat-completions/HCX-003',
                           headers=headers, json=completion_request, stream=True) as r:
            for line in r.iter_lines():
                if line:
                    ret.append(line.decode("utf-8"))
        return ret

def show_document_details(selected_date):
    selected_docs = merge_df[merge_df['date'] == selected_date]

    if not selected_docs.empty:
        for index, row in selected_docs.iterrows():
            title = row['title']
            st.subheader(title)

            polarity_score = row['tone_doc']
            tone_label = 'Hawkish' if polarity_score > 0 else 'Dovish'
            st.subheader(f"Polarity Score: {polarity_score} ({tone_label})")

            col1, col2 = st.columns(2)
            content = row['content'].split('\n')
            col1.text_area('Content', '\n'.join(content[:5]), height=300)

            completion_executor = CompletionExecutor(
                host='https://clovastudio.stream.ntruss.com',
                api_key='NTA0MjU2MWZlZTcxNDJiY6o7O0mMGUuTEHU6yLaRpv/2IkicvAMe/Pab0BKS5gW8',
                api_key_primary_val='gIAB8vXgHEn5ZwAEgBHbnj6qZVa45KdMxz85pTjT',
                request_id='85faed7a-d6fc-413b-8858-513aeaebe9f1'
            )

            col2.write('금통위 의사록 내용 요약')

            minutes = row['content'][:500]

            if minutes:
                preset_text = [{"role": "system", "content": "- 데이터를 해독하고, 파싱하여 핵심 내용을 추출합니다."},
                               {"role": "user", "content": minutes}]

                request_data = {
                    'messages': preset_text,
                    'topP': 0.6,
                    'topK': 0,
                    'maxTokens': 500,
                    'temperature': 0.1,
                    'repeatPenalty': 1.2,
                    'stopBefore': [],
                    'includeAiFilters': True,
                    'seed': 0
                }

                result = completion_executor.execute(request_data)
                try:
                    result_text = json.loads(result[-4][5:])['message']['content']
                    col2.write(result_text)
                except (IndexError, KeyError, json.JSONDecodeError) as e:
                    col2.error("Error in processing the response from the API")

            ngrams = row['ngrams'].split('\n')
            col1.text_area(f'Ngrams', '\n'.join(ngrams[:5]), height=330)

            # Show h_cnt and d_cnt using Plotly bar chart
            h_cnt = row['h_cnt']
            d_cnt = row['d_cnt']

            fig = go.Figure()

            # 오른쪽 열: h_cnt와 d_cnt 막대 그래프
            fig.add_trace(go.Bar(x=['Hawkish', 'Dovish'], y=[h_cnt, d_cnt],
                                 marker_color=['red', 'blue']))  # h_cnt를 빨간색으로 설정

            fig.update_layout(title=f'Hawkish vs Dovish Count',
                              xaxis_title='Sentiment', yaxis_title='Count',
                              showlegend=False,
                              width=800, height=400)

            # Plotly 차트를 Streamlit에 표시
            col2.plotly_chart(fig)
    else:
        st.warning(f"No documents found for {selected_date}. Please select another date.")

if __name__ == '__main__':
    # Streamlit code to select a date
    dates = merge_df['date'].unique()
    selected_date = st.sidebar.selectbox('Select a date', dates)

    if selected_date:
        show_document_details(selected_date)

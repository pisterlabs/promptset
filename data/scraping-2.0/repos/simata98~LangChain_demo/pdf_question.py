import tiktoken
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQA
import streamlit as st
import tempfile
import os
from streamlit_extras.buy_me_a_coffee import button
from streamlit_extras.let_it_rain import rain
from streamlit_extras.metric_cards import style_metric_cards
from google.cloud import firestore
import datetime
import pytz

# 한국 시간대 설정
korea_tz = pytz.timezone('Asia/Seoul')

# FireStore 연결 및 값 불러오기
db = firestore.Client.from_service_account_json('llm-service-9990d-firebase-adminsdk-kzzo8-875179b253.json')
today = datetime.datetime.now(tz=korea_tz).date()
today_date_str = today.strftime("%Y-%m-%d")
yesterday_date_str = (today - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
today_count_ref = db.collection('daily_counts').document(today_date_str)
yesterday_date_ref = db.collection('daily_counts').document(yesterday_date_str)
today_count_doc = today_count_ref.get()
yesterday_count_doc = yesterday_date_ref.get()
today_visit_ref = db.collection('daily_counts').document(today_date_str)
yesterday_visit_ref = db.collection('daily_counts').document(yesterday_date_str)
today_visit_doc = today_visit_ref.get()
yesterday_visit_doc = yesterday_visit_ref.get()

if not today_count_doc.exists:
    usage_count = 0
    today_count_ref.set({'usage_count': usage_count, 'visit_count': 0})
    print(f'daily_count created : {today_date_str}')
else:
    usage_count = today_count_doc.get('usage_count')
    today_visit_ref.update({'visit_count': today_visit_doc.get('visit_count') + 1})
    print(f'daily_count exists : {today_date_str}')


def rain_drop():
    rain(
        emoji="💻",
        font_size=32,
        falling_speed=10,
        animation_length=0.5,
    )


# 사용 현황 업데이트
def metric_card():
    col1, col2 = st.columns(2)
    col1.metric(label="총 실행횟수",
                value=today_count_doc.get('usage_count'),
                delta=today_count_doc.get('usage_count') - yesterday_count_doc.get('usage_count'))
    col2.metric(label="총 방문횟수",
                value=today_count_doc.get('visit_count'),
                delta=today_count_doc.get('visit_count') - yesterday_count_doc.get('visit_count'))
    style_metric_cards(background_color='#5fdcde')


def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_tilepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_tilepath, 'wb') as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_tilepath)
    pages = loader.load_and_split()
    return pages


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


# streamlit
button(username="c4nd0it", floating=True, width=221)
rain_drop()
st.title('KT 인공지능 서비스')
st.write('KT 인공지능 서비스는 _LLM_ 과 _LangChain_ 을 활용하여 만들어졌습니다.')
metric_card()
st.write("---")
# OpenAi 키 받기
openai_key = st.text_input("OPEN_AI_API_KEY", type="password")
# upload file
uploaded_file = st.file_uploader("PDF파일을 업로드 해주세요", type=['pdf'])
st.write("---")

# LLM 실행
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)
    # Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.split_documents(pages)

    # Embedding
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_key)

    # load it into chroma
    db = Chroma.from_documents(texts, embeddings_model)

    st.header('PDF에게 질문해보세요!')
    question = st.text_input('질문을 입력하세요')

    if st.button('질문하기'):
        current_count = today_count_doc.get('usage_count')
        new_count = current_count + 1
        today_count_ref.update({'usage_count': new_count})
        with st.spinner('Wait for it...'):
            chat_box = st.empty()
            stream_handler = StreamHandler(chat_box)
            llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                             temperature=0,
                             openai_api_key=openai_key,
                             streaming=True, callbacks=[stream_handler])
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
            result = qa_chain({"query": question})


# db 저장 후 임베딩 불러오기
# https://python.langchain.com/docs/integrations/vectorstores/chroma

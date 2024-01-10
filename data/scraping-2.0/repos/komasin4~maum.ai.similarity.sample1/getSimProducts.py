from fuzzywuzzy import fuzz
import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import numpy as np
from openai import AzureOpenAI
from Levenshtein import distance as levenshtein_distance
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
from streamlit.logger import get_logger

load_dotenv(verbose=True)

logger = get_logger(__name__)

def read_product_list(file_path):
    try:
        data = pd.read_csv(file_path)
        # print("product list", data)
        return data
    except Exception as e:
        print(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return None
    
#def extract_product_name(question):


def call_openai_chat(type, question):
    logger.info("call_openai_chat:" + type + ":" + question + ":")
    client = AzureOpenAI(
        api_key = os.getenv("AZURE_OPENAI_KEY"),  
        api_version = os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") 
    )
    
    if(type == 'product'):
        content = "질문에서 제품명을 추출합니다. 특정 요리 이름도 제품명으로 간주합니다. 제품명만 말해줍니다. 제품명 추출이 어려울때는 '제품없음' 이라고 답변합니다."
    else:
        content = "당신은 대한민국 식품회사 풀무원의 제품 담당자 입니다. 사용자의 제품문의에 대한 답을 합니다."
    try:
        response = client.chat.completions.create(
            # model="gpt-3.5-turbo",
            model="gpt-4",
            messages=[
                #{"role": "system", "content": "질문에서 제품명을 추출합니다."},
                {"role": "system", "content": content},
                {"role": "user", "content": question}
            ]
        )
        product_name = response.choices[0].message.content.strip()
        return product_name
    except Exception as e:
        print(f"OpenAI GPT-3 요청 중 오류가 발생했습니다: {e}")
        return None

# Levenshtein Distance Algorithm
def find_similar_products(user_query, product_data):
    # Levenshtein 거리를 계산하여 가장 가까운 제품을 찾음
    product_data['Similarity'] = product_data['제품명'].apply(
        lambda x: levenshtein_distance(user_query, x))
    sorted_products = product_data.sort_values('Similarity')
    return sorted_products.head(3).to_string(index=False)

# Hamming Distance
def find_similar_products_hamming(user_query, product_data):
    # Hamming Distance를 이용하여 가장 유사한 제품 찾기
    # Hamming Distance는 두 문자열의 길이가 동일해야 하므로, 길이를 맞춤
    max_length = max(len(user_query), max(len(prod)
                     for prod in product_data['제품명']))
    user_query_padded = user_query.ljust(max_length)

    product_data['Similarity'] = product_data['제품명'].apply(
        lambda x: sum(ch1 != ch2 for ch1, ch2 in zip(
            x.ljust(max_length), user_query_padded)) / max_length
    )

    # 유사도(낮은 값이 더 유사함)에 따라 제품들 정렬
    sorted_products = product_data.sort_values('Similarity')

    return sorted_products.head(3).to_string(index=False)
    #return sorted_products[['구분', '대분류', 'NO', '제품명', 'Similarity']].head(3).to_string(index=False)

# FuzzyWuzzy
def find_similar_products_fuzzywuzzy(user_query, product_data):
    # FuzzyWuzzy를 이용하여 가장 유사한 제품 찾기
    product_data['Similarity'] = product_data['제품명'].apply(
        lambda x: fuzz.ratio(user_query, x)
    )

    # 유사도(높은 값이 더 유사함)에 따라 제품들 정렬
    sorted_products = product_data.sort_values('Similarity', ascending=False)

    #return sorted_products[['구분', '대분류', 'NO', '제품명', 'Similarity']].head(3).to_string(index=False)
    return sorted_products.head(3).to_string(index=False)
    
# N-gram Analysis
def ngram_similarity(text1, text2, n=2):
    """
    두 문자열 간의 n-gram 기반 코사인 유사도를 계산합니다.
    """
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
    ngrams = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(ngrams)[0, 1]

def find_similar_products_ngram(user_query, product_data, n=2):
    """
    n-gram 유사도를 이용하여 가장 유사한 제품을 찾습니다.
    """
    product_data['Similarity'] = product_data['제품명'].apply(
        lambda x: ngram_similarity(user_query, x, n=n)
    )

    # 유사도(높은 값이 더 유사함)에 따라 제품들 정렬
    sorted_products = product_data.sort_values('Similarity', ascending=False)

    #return sorted_products[['구분', '대분류', 'NO', '제품명', 'Similarity']].head(3).to_string(index=False)
    return sorted_products.head(3).to_string(index=False)

# Sequence Matching Algorithms 
def find_similar_products_sequence_matching(user_query, product_data):
    """
    Sequence Matching 알고리즘을 이용하여 가장 유사한 제품을 찾는 함수.
    """
    # 유사도 계산을 위한 함수 정의
    def calculate_similarity(product_name):
        # Sequence Matching 알고리즘을 사용하여 유사도 계산
        return SequenceMatcher(None, user_query, product_name).ratio()

    # 각 제품에 대해 유사도 계산
    product_data['Similarity'] = product_data['제품명'].apply(
        calculate_similarity)

    # 유사도가 0.3 이상인 제품만 필터링
    filtered_products = product_data[product_data['Similarity'] >= 0.3]

    # 유사도에 따라 제품들을 정렬
    #sorted_products = product_data.sort_values(
    sorted_products = filtered_products.sort_values(
        by='Similarity', ascending=False)

    # 상위 3개 제품 반환
    return sorted_products.head(3)


def printPrds():
    products = ""
    for index, row in product_data.iterrows():
        products += row.to_string(index=False).replace(" ",
                                                    "").replace("\n", ".") + "\n"

    st.text_area('참고 - 제품목록', products, height=500)    


if 'prev_question' not in st.session_state:
    st.session_state.prev_question = ''
if 'product_name' not in st.session_state:
    st.session_state.product_name = ''
if 'action_step' not in st.session_state:
    st.session_state.action_step = '0'
if 'similar_products' not in st.session_state:
    st.session_state.similar_products = []
if 'selected_item' not in st.session_state:
    st.session_state.selected_item = ''
if 'item_list' not in st.session_state:
    st.session_state.item_list = []
if 'desc_list' not in st.session_state:
    st.session_state.desc_list = []

print("1.st.session_state.action_step", st.session_state.action_step)

# 제품 목록 읽기
file_path = "/Users/komasin4/Data/maum.ai/pulmuwon/mergelist_1.csv"
product_data = read_product_list(file_path)

#화면
st.title('유사제품 추출 테스트')

# question = st.text_input('질문', placeholder='질문을 입력하세요.')
question = st.text_input('질문내용입력', '제주 보리차는 얼마인가요?')

#al = st.radio(
#    "알고리즘 선택",
#    ["Levenshtein Distance", "Hamming Distance", "FuzzyWuzzy", "N-gram Analysis", "Sequence Matching"])
#    #captions=["Laugh out loud.", "Get the popcorn.", "Never stop learning."])

if st.button('물어보기!!!'):
    print("if-1")
    
    print("1|", question, "|", st.session_state.prev_question)
    
    with st.spinner('기다려주세요...'):
        if st.session_state.prev_question != question:
            print("extract product name...")
            st.session_state.product_name = call_openai_chat('product', question)
            st.session_state.prev_question = question
        st.write("제품명 : ", st.session_state.product_name)
        st.session_state.similar_products = find_similar_products_sequence_matching(
            st.session_state.product_name, product_data)

        print("size:", st.session_state.similar_products.size)

        if st.session_state.similar_products.size < 1:
            st.write("당사에 없는 제품인것 같습니다. 제품명을 다시 확인해 주세요.")
        else:
            st.session_state.item_list = list()
            st.session_state.desc_list = list()

            for index, row in st.session_state.similar_products.iterrows():
                st.session_state.item_list.append(row.제품명)
                st.session_state.desc_list.append("순번:" + str(row.NO) +
                            ", 유사도:" + str(row.Similarity))

            st.session_state.selected_item = st.radio(
                "제품을 선택하세요", st.session_state.item_list, captions=st.session_state.desc_list, index=None)
            
            print("st.session_state.selected_item", st.session_state.selected_item)

            print("2|", question, "|", st.session_state.prev_question)
            
        st.session_state.action_step = 1
elif st.session_state.action_step == 1:
    print("if-2")
    #if st.session_state.prev_question != question:
    #    st.session_state.product_name = extract_product_name(question)
    st.write("제품명 : ", st.session_state.product_name)
    st.session_state.similar_products = find_similar_products_sequence_matching(
        st.session_state.product_name, product_data)
    #st.session_state.prev_question = question
    
    if len(st.session_state.item_list) > 0:
        st.session_state.selected_item = st.radio(
            "제품을 선택하세요", st.session_state.item_list, captions=st.session_state.desc_list, index=None)

        print("st.session_state.selected_item", st.session_state.selected_item)
        print(st.session_state.product_name, "->", st.session_state.selected_item)

        if st.session_state.selected_item is not None:
            new_question = question.replace(
                st.session_state.product_name, st.session_state.selected_item)

            st.write("수정질문: ", new_question)
    
    #with st.spinner('제품에 대해 문의 중입니다. 잠시만 기다려주세요...'):
    #    answer = call_openai_chat('question', new_question)
    #    st.write("답변")
    #    st.write(answer)

    print("3|", question, "|", st.session_state.prev_question)

    

printPrds()
print("2.st.session_state.action_step", st.session_state.action_step)





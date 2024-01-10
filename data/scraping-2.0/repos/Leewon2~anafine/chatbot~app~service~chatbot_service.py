import os
import openai
from dotenv import load_dotenv
# .env 파일 불러오기
load_dotenv()

# 환경 변수 사용하기
openai_api_key = os.getenv('OPENAI_API_KEY')
serpapi_api_key = os.getenv('SERPAPI_API_KEY')
papago_client_id = os.getenv('PAPAGO_CLIENT_ID')
papago_client_secret = os.getenv('PAPAGO_CLIENT_SECRET')

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from transformers import pipeline
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.chains.summarize import load_summarize_chain
import textwrap

llm = OpenAI(temperature=0)

tools = load_tools(["serpapi", "llm-math"], llm=llm)

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

import requests

def translate_text(text, source_lang, target_lang, client_id, client_secret):
    url = "https://openapi.naver.com/v1/papago/n2mt"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret
    }
    data = {
        "source": source_lang,
        "target": target_lang,
        "text": text
    }
    
    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        translated_text = response.json().get('message', {}).get('result', {}).get('translatedText', '')
        return translated_text
    else:
        print("Error Code:", response.status_code)
        return None

def get_search_results(query, api_key, language="en"):
    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "hl": language
    }
    
    try:
        response = requests.get("https://serpapi.com/search", params=params)
        response.raise_for_status()
        search_results = response.json()
        return search_results
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"An error occurred: {err}")
    return None

def preprocess_prompt(prompt):
    # 모델에 입력할 수 있는 토큰의 최대 수
    MAX_TOKENS = 8000
    # prompt를 토큰으로 변환 (공백 기준으로 나눔)
    tokens = prompt.split()
    # 토큰의 길이가 최대 토큰 수를 초과하는지 확인
    if len(tokens) > MAX_TOKENS:
        # 초과한다면 최대 길이에 맞춰 줄임
        return ' '.join(tokens[:MAX_TOKENS])
    else:
        # 초과하지 않는다면 그대로 반환
        return prompt



class ChatService:
    def play_normal_chat(self, data):
        user_input = data.get('message', '')

        # OpenAI 채팅 모델을 사용하여 대답 생성
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # 채팅에 최적화된 모델
            messages=[
                {"role": "user", "content": user_input}
            ],
            max_tokens=150
        )

        # 생성된 응답의 텍스트 추출
        gpt_response = response.choices[0].message['content'].strip()

        return {'response': gpt_response}
    def play_chat(self,data):
        #사용자 성별 입력 받기
        user_sex = data['sex']
        # 사용자 나이 입력 받기
        user_age = data['age']
        # 아픈 부위 입력 받기
        user_pain_area = data['pain_area']
        # 증상 입력 받기
        user_symptoms = data['symptoms']


        # 원인과, 의심되는 병으로 나누어서 구글 검색
        user_input_cause = f"{user_age}세, {user_sex},  {user_pain_area},  {user_symptoms}, 원인"
        user_input_disease = f"{user_age}세, {user_sex},  {user_pain_area},  {user_symptoms}, 의심 병"
        user_input_doubt = f"{user_age}세, {user_sex},  {user_pain_area},  {user_symptoms}, 일 때 받아야 할 검사"

        translated_input_cause = translate_text(user_input_cause, "ko", "en", papago_client_id, papago_client_secret)
        translated_input_disease = translate_text(user_input_disease, "ko", "en", papago_client_id, papago_client_secret)
        translated_input_doubt = translate_text(user_input_doubt, "ko", "en", papago_client_id, papago_client_secret)

        # 영어로 검색하여 응답 받기
        english_search_results_cause = get_search_results(translated_input_cause, serpapi_api_key, language="en")
        english_search_results_disease = get_search_results(translated_input_disease, serpapi_api_key, language="en")
        english_search_results_doubt = get_search_results(translated_input_doubt, serpapi_api_key, language="en")

        # 한국어로 검색하여 응답 받기
        korean_search_results_cause = get_search_results(user_input_cause, serpapi_api_key, language="ko")
        korean_search_results_disease = get_search_results(user_input_disease, serpapi_api_key, language="ko")
        korean_search_results_doubt = get_search_results(user_input_doubt, serpapi_api_key, language="ko")

        # 영어 검색 결과에서 organic_results 부분 추출
        english_results_text_cause = ' '.join([result['snippet'] for result in english_search_results_cause.get('organic_results', [])])
        english_results_text_disease = ' '.join([result['snippet'] for result in english_search_results_disease.get('organic_results', [])])
        english_results_text_doubt = ' '.join([result['snippet'] for result in english_search_results_doubt.get('organic_results', [])])

        # 한국어 검색 결과에서 'organic_results' 부분 추출
        korean_results_text_cause = ' '.join([result['snippet'] for result in korean_search_results_cause.get('organic_results', [])])
        korean_results_text_disease = ' '.join([result['snippet'] for result in korean_search_results_disease.get('organic_results', [])])
        korean_results_text_doubt = ' '.join([result['snippet'] for result in korean_search_results_doubt.get('organic_results', [])])


        # Summary 템플릿 정의
        prompt_template_english_results_text_cause = """
        summary {english_results_text_cause}, in a sentence
        Write a concise bullet point summary of the following:
        {text}

        CONCISE SUMMARY IN BULLET POINTS:
        """

        prompt_template_english_results_text_disease = """
        summary {english_results_text_disease}, in a sentence
        Write a concise bullet point summary of the following:
        {text}

        CONCISE SUMMARY IN BULLET POINTS:
        """

        prompt_template_english_results_text_doubt = """
        summary {english_results_text_doubt}, in a sentence
        Write a concise bullet point summary of the following:
        {text}

        CONCISE SUMMARY IN BULLET POINTS:
        """

        prompt_template_korean_results_text_cause= """
        summary {korean_results_text_cause}, in a sentence
        Write a concise bullet point summary of the following:
        {text}

        CONCISE SUMMARY IN BULLET POINTS:
        """

        prompt_template_korean_results_text_disease = """
        summary {korean_results_text_disease} in a sentence
        Write a concise bullet point summary of the following:
        {text}

        CONCISE SUMMARY IN BULLET POINTS:
        """

        prompt_template_korean_results_text_doubt = """
        summary {korean_results_text_doubt} in a sentence
        Write a concise bullet point summary of the following:
        {text}

        CONCISE SUMMARY IN BULLET POINTS:
        """


        # Text Splitter 및 Summarization Chain 초기화
        text_splitter = CharacterTextSplitter()
        english_results_text_cause_docs = text_splitter.create_documents(english_results_text_cause)[:4]
        english_results_text_disease_docs = text_splitter.create_documents(english_results_text_disease)[:4]
        english_results_text_doubt_docs = text_splitter.create_documents(english_results_text_doubt)[:4]
        korean_results_text_cause_docs = text_splitter.create_documents(korean_results_text_cause)[:4]
        korean_results_text_disease_docs = text_splitter.create_documents(korean_results_text_disease)[:4]
        korean_results_text_doubt_docs = text_splitter.create_documents(korean_results_text_doubt)[:4]

        BULLET_POINT_PROMPT_english_results_text_cause = PromptTemplate(template=prompt_template_english_results_text_cause, input_variables=["english_results_text_cause", "text"])
        BULLET_POINT_PROMPT_english_results_text_disease = PromptTemplate(template=prompt_template_english_results_text_disease, input_variables=["english_results_text_disease","text"])
        BULLET_POINT_PROMPT_english_results_text_doubt = PromptTemplate(template=prompt_template_english_results_text_doubt, input_variables=["english_results_text_doubt","text"])
        BULLET_POINT_PROMPT_korean_results_text_cause = PromptTemplate(template=prompt_template_korean_results_text_cause, input_variables=["korean_results_text_cause","text"])
        BULLET_POINT_PROMPT_korean_results_text_disease = PromptTemplate(template=prompt_template_korean_results_text_disease, input_variables=["korean_results_text_disease","text"])
        BULLET_POINT_PROMPT_korean_results_text_doubt = PromptTemplate(template=prompt_template_korean_results_text_doubt, input_variables=["korean_results_text_doubt","text"])
            
        chain_english_results_text_cause= load_summarize_chain(llm, chain_type="stuff", prompt=BULLET_POINT_PROMPT_english_results_text_cause)
        chain_english_results_text_disease = load_summarize_chain(llm, chain_type="stuff", prompt=BULLET_POINT_PROMPT_english_results_text_disease)
        chain_english_results_text_doubt = load_summarize_chain(llm, chain_type="stuff", prompt=BULLET_POINT_PROMPT_english_results_text_doubt)
        chain_korean_results_text_cause = load_summarize_chain(llm, chain_type="stuff", prompt=BULLET_POINT_PROMPT_korean_results_text_cause)
        chain_korean_results_text_disease = load_summarize_chain(llm, chain_type="stuff", prompt=BULLET_POINT_PROMPT_korean_results_text_disease)
        chain_korean_results_text_doubt = load_summarize_chain(llm, chain_type="stuff", prompt=BULLET_POINT_PROMPT_korean_results_text_doubt)


        # Summarization 실행
        output_summary_english_results_text_cause = chain_english_results_text_cause.run({'english_results_text_cause': english_results_text_cause, 'input_documents': english_results_text_cause_docs})
        output_summary_english_results_text_disease = chain_english_results_text_disease.run({'english_results_text_disease': english_results_text_disease, 'input_documents': english_results_text_disease_docs})
        output_summary_english_results_text_doubt = chain_english_results_text_doubt.run({'english_results_text_doubt': english_results_text_doubt, 'input_documents': english_results_text_doubt_docs})
        output_summary_korean_results_text_cause = chain_korean_results_text_cause.run({'korean_results_text_cause': korean_results_text_cause, 'input_documents': korean_results_text_cause_docs})
        output_summary_korean_results_text_disease = chain_korean_results_text_disease.run({'korean_results_text_disease': korean_results_text_disease, 'input_documents': korean_results_text_disease_docs})
        output_summary_korean_results_text_doubt = chain_korean_results_text_doubt.run({'korean_results_text_doubt': korean_results_text_doubt, 'input_documents': korean_results_text_doubt_docs})

        # 결과 출력
        wrapped_text_english_results_text_cause = textwrap.fill(output_summary_english_results_text_cause, width=100, break_long_words=False, replace_whitespace=False)
        wrapped_text_english_results_text_disease = textwrap.fill(output_summary_english_results_text_disease, width=100, break_long_words=False, replace_whitespace=False)
        wrapped_text_english_results_text_doubt = textwrap.fill(output_summary_english_results_text_doubt, width=100, break_long_words=False, replace_whitespace=False)
        wrapped_text_korean_results_text_cause = textwrap.fill(output_summary_korean_results_text_cause, width=100, break_long_words=False, replace_whitespace=False)
        wrapped_text_korean_results_text_disease = textwrap.fill(output_summary_korean_results_text_disease, width=100, break_long_words=False, replace_whitespace=False)
        wrapped_text_korean_results_text_doubt = textwrap.fill(output_summary_korean_results_text_doubt, width=100, break_long_words=False, replace_whitespace=False)

        # 검색 결과를 통합 후 LLM에게 전달
        combined_results_disease = wrapped_text_english_results_text_disease + " " + wrapped_text_korean_results_text_disease 
        preprocessed_prompt_disease = preprocess_prompt(combined_results_disease)
        disease_prompt = f"이 증상으로 인한 예상되는 질병은 무엇입니까? \n\n{preprocessed_prompt_disease} \n\모든 답변은 한국어로 해주세요."
        llm_response_disease = agent.run(input=disease_prompt, max_tokens=3000)

        combined_results_cause = wrapped_text_english_results_text_cause + " " +  wrapped_text_korean_results_text_cause
        preprocessed_prompt_cause = preprocess_prompt(combined_results_cause)
        cause_prompt = f"이 증상의 예상되는 원인은 무엇입니까? \n\n{preprocessed_prompt_cause} \n\n모든 답변은 한국어로 해주세요."
        llm_response_cause = agent.run(input=cause_prompt, max_tokens=3000)


        combined_results_doubt = wrapped_text_english_results_text_doubt + " " +  wrapped_text_korean_results_text_doubt
        preprocessed_prompt_doubt = preprocess_prompt(combined_results_doubt)
        doubt_prompt = f"어떤 검사를 받아보는 것이 좋을까요? 어떤 검사를 받으면 좋을지 추천 검사만 알려주세요. \n\n{preprocessed_prompt_doubt} \n\n 모든 답변은 한국어로 해주세요."
        llm_response_doubt = agent.run(input=doubt_prompt, max_tokens=3000)

            
        # 최종적으로 분석된 결과 출력
        return {
            'disease': llm_response_disease,
            'cause': llm_response_cause,
            'recommended_tests': llm_response_doubt
        }
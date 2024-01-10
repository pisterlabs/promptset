from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import ChatPromptTemplate

# 키 관리는 철저히
import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_KEY = os.environ.get('OPENAI_KEY')

question = '사과의 발언에 대해 정리해 줄래?'

# index_file에서 불러오기
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
knowledge_base = FAISS.load_local("faiss_index", embeddings)


docs = knowledge_base.similarity_search(question)  # 코사인 유사도가 아닌 FAISS가 제공하는 유사도 알고리즘, score가 0~1사이의 값을 가지지 않음
# print(len(docs))  # 4가 나옴
# docs = docs[:2]  # 토큰 절약

# 질문하기
llm = ChatOpenAI(
        # temperature=0,
        temperature=0.2,
        openai_api_key=OPENAI_KEY,
        max_tokens=2000,
        # model_name='gpt-3.5-turbo-1106',  # 최신이지만 답변이 더 딱딱함...
        # model_name='gpt-3.5-turbo',
        model_name='gpt-3.5-turbo-16k',  
        request_timeout=120
        )

chat_template = ChatPromptTemplate.from_messages(
    [
        # ("system", "너의 이름은 자비스. 너는 충실한 보조 AI이고 선생인 내 일을 돕고 질문에 답할거야. 너는 애교를 부리며 학생답고 귀엽게 답변해야 하고, 답변은 5줄 이하여야 해."),
        # ("system", "I am teacher and You are a faithful assistant AI. Your name is 자비스"),
        ("human", "{user_input}"),
    ]
)

messages = chat_template.format_messages(user_input=question)
chain = load_qa_chain(llm, chain_type="stuff")

response = chain.run(input_documents=docs, question=messages)

# 원하지 않은 형태의 답변 타입 제거
response.replace('자비스: ', '')
response.replace('자비스:', '')

print('질문 :', question)
print('답변 :', response)
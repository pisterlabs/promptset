from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import time

# 키 관리는 철저히
import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_KEY = os.environ.get('OPENAI_KEY')

# 파일 읽기
file_path = 'summary.txt'
file_content = ''
with open(file_path, 'r', encoding='utf-8') as file:
    file_content = file.read()

# chunk 분할
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=500,  # 거의 토큰값과 유사. 3.5-turbo인 경우 4096 토큰(16k면 4배) / 우리는 chunk 4개와 prompt와 질문을 보내야 함...!
    chunk_overlap=100,  # chunk끼리 약간 겹치게 해야, 내용이 이어짐
    length_function=len
)
chunks = text_splitter.split_text(file_content)

# ratelimit를 고려한 chunk 분리
# print(len(chunks)) # chunk의 길이를 보고 1000단위로 나누기로 결정
chunks1 = chunks[:1000]
chunks2 = chunks[1000:2000]
chunks3 = chunks[2000:3000]
chunks4 = chunks[3000:4000]
chunks5 = chunks[4000:5000]
chunks6 = chunks[5000:6000]
chunks7 = chunks[6000:]

# 각각 임베딩 학습
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
knowledge_base1 = FAISS.from_texts(chunks1, embeddings)
print('1/7 embedded')
time.sleep(60)
knowledge_base2 = FAISS.from_texts(chunks2, embeddings)
print('2/7 embedded')
time.sleep(60)
knowledge_base3 = FAISS.from_texts(chunks3, embeddings)
print('3/7 embedded')
time.sleep(60)
knowledge_base4 = FAISS.from_texts(chunks4, embeddings)
print('4/7 embedded')
time.sleep(60)
knowledge_base5 = FAISS.from_texts(chunks5, embeddings)
print('5/7 embedded')
time.sleep(60)
knowledge_base6 = FAISS.from_texts(chunks6, embeddings)
print('6/7 embedded')
time.sleep(60)
knowledge_base7 = FAISS.from_texts(chunks7, embeddings)
print('7/7 embedded')

# 임베딩 결과 각각 저장
knowledge_base1.save_local("faiss_index1")
knowledge_base2.save_local("faiss_index2")
knowledge_base3.save_local("faiss_index3")
knowledge_base4.save_local("faiss_index4")
knowledge_base5.save_local("faiss_index5")
knowledge_base6.save_local("faiss_index6")
knowledge_base7.save_local("faiss_index7")

# 결과갑 머지
knowledge_base1.merge_from(knowledge_base2)
print('1/6 merging')
knowledge_base1.merge_from(knowledge_base3)
print('2/6 merging')
knowledge_base1.merge_from(knowledge_base4)
print('3/6 merging')
knowledge_base1.merge_from(knowledge_base5)
print('4/6 merging')
knowledge_base1.merge_from(knowledge_base6)
print('5/6 merging')
knowledge_base1.merge_from(knowledge_base7)
print('6/6 merging')

# 임베딩 결과 최종 저장
knowledge_base1.save_local("faiss_index")

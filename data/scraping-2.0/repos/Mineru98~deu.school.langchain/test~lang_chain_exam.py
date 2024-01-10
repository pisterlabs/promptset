# -*- coding:utf-8 -*-
import os
import re
import sys
import time
import itertools
import pandas as pd
from glob import glob
from typing import List
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain import OpenAI, PromptTemplate
from langchain.agents import create_csv_agent
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings

path = os.path.dirname(os.path.abspath(__file__))
SCR_PATH = os.path.abspath(__file__)
SCR_DIR, SCR_BN = os.path.split(SCR_PATH)
REPO_DIR = os.path.abspath(SCR_DIR + "/..")
ASSETS_DIR = os.path.abspath(REPO_DIR + "/static")
csv_list = glob(ASSETS_DIR + "/*.csv")

templates = {
    "case1": """'{input_text}'라는 질문에 대한 대답은 '{output_text}'입니다. 이를 안내하듯이 부드러운 말투로 대답해주세요. 대답할 때는 줄임말을 쓰지 않아야 합니다. """,
    "case2": """ChatGPT 모델을 다음 지침을 따르세요.
1. 대답을 할 때는 부드러운 말투로 대답해야 합니다.
2. 대답을 할 때는 안내하듯이 대답해야 합니다.
3. 반드시 한국어로 대답해야합니다.
4. 당신은 동의대학교에 대한 정보만 대답해야 합니다.
5. '하지만'과 같은 대답을 해서는 안됩니다.
6. 모든 대답에서는 출처를 남겨줘야합니다.
7. 출처를 반드시 알려줘야 합니다.
---
'{input_text}'라는 질문에 대한 대답은 '{output_text}'입니다.""",
}

try:
    load_dotenv()
    apikey = os.environ["OPENAI_API_KEY"]
except:
    print("Please, export your OpenAI API KEY over 'OPENAI_API_KEY' environment variable")
    print("You may create the key here: https://platform.openai.com/account/api-keys")
    sys.exit(1)


def slow_print(text):
    count = 0
    for char in text:
        print(char, end="", flush=True)
        if char in [".", "?"]:
            time.sleep(0.5)
        elif count % 5 == 0:
            time.sleep(0.2)
        else:
            time.sleep(0.05)
        count += 1
    print()


def sum_dataframe_lengths(df):
    total_length = 0
    for column in df.columns:
        column_length = df[column].astype(str).apply(len).sum()
        total_length += column_length
    return total_length


def csv_parser(query, filePath):
    df = pd.read_csv(filePath)
    if sum_dataframe_lengths(df) > 1000:
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k", max_tokens=9000)
    else:
        llm = OpenAI(temperature=0, model_name="text-davinci-003")
    try:
        agent = create_csv_agent(llm, filePath, verbose=True, kwargs={"max_iterations": 3})
        res = agent.run(query)
        return res
    except Exception as e:
        print(e)
        return "찾을 수 없습니다."


def load_files(query) -> List[Document]:
    embeddings = OpenAIEmbeddings()
    docs = [
        CSVLoader(file, csv_args={"delimiter": ",", "quotechar": '"'}).load()
        for file in filter(lambda x: x.endswith("menual.csv"), csv_list)
    ]
    docs = list(itertools.chain(*docs))
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 1})
    docs = retriever.get_relevant_documents(query)
    return docs


slow_print("동의대학교 학사 정보 시스템 챗봇 DEU GPT입니다.\n무엇이 궁금하시나요? (종료를 하려면, 0 입력)")
while True:
    query = input(":")
    time.sleep(1)
    if query == "0":
        slow_print("다음에 또 만나요😀")
        sys.exit(0)
    elif len(query) == 0:
        slow_print("아무것도 궁금하지 않으신가봐요?")
        continue
    elif len(query) == 100:
        slow_print("100자 이내로 질문해주세요.")
        continue
    slow_print("찾는 중...")
    res = load_files(query)
    if len(res) > 0:
        pattern = r"filename: (\S+\.csv)"
        match = re.search(pattern, res[0].page_content.split("\n")[0])
        fileName = match.group(1)
        res = csv_parser(query=query, filePath=(ASSETS_DIR + "/" + fileName))
        prompt = PromptTemplate(
            input_variables=["input_text", "output_text"],
            template=templates["case2"],
        )
        texts = prompt.format_prompt(input_text=query, output_text=res)
        model = OpenAI(model_name="text-davinci-003", temperature=0.0)
        result = model(texts.to_string())
        slow_print(result)
    else:
        slow_print("현재 저에게는 관련된 자료를 찾을 수 없습니다.😭\n해당 내용은 업데이트 할 수 있도록 하겠습니다.")

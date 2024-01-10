from pathlib import Path
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser, output_parser
from langchain.output_parsers import CommaSeparatedListOutputParser
import numpy as np
import pandas as pd
import openai
import os
import json
from .models import Word, Quiz
import re

BASE_DIR = Path(__file__).resolve().parent.parent

with open(BASE_DIR/'secrets.json') as f:
    secrets = json.loads(f.read())

os.environ['OPENAI_API_KEY']= secrets['OPENAI_API_KEY']

# GPT AI를 활용한 문장 생성하기
def make_sentence(word, meaning):
    sentence_prompt = ChatPromptTemplate.from_messages([("system", """ Make one sentence using '{word}' and only korean.""",)])
    sentence_parser = CommaSeparatedListOutputParser()

    llm = ChatOpenAI(
        temperature=1,
        # model="ft:gpt-3.5-turbo-0613:personal::8aG6fMiE",
        # model="ft:gpt-3.5-turbo-1106:personal::8aztbxTn",
        model="gpt-3.5-turbo-1106",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )

    sentence_chain = sentence_prompt | llm | sentence_parser

    # 문장 만들기
    temp=sentence_chain.invoke({"word":word})
    # ','으로 끊어지는 문제 해결    
    sentence=temp[0]
    for i in range(1,len(temp)):
        sentence+=(', '+temp[i])

    result = {
        "sentence": sentence
    }
    

    return result
    

# 문제 만들기
def make_problem(word, meaning):
    # 파인튜닝 모델
    llm = ChatOpenAI(
        temperature=0.1,
        #model="ft:gpt-3.5-turbo-0613:personal::8aG6fMiE",
        #model="ft:gpt-3.5-turbo-1106:personal::8aztbxTn",
        model="gpt-3.5-turbo-1106",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )
    
    # 문장 생성 프롬프트(1개 생성)
    sent_prompt = ChatPromptTemplate.from_messages([("system", """ Make one sentence using '{word}' and only korean.""",)])
    
    # 생성된 문장 List로 변환
    sentence_parser = CommaSeparatedListOutputParser()
    
    # 체인 생성
    sent_chain = sent_prompt | llm | sentence_parser
    
    # 임시 저장소
    result=dict()
    
    # 문장 만들기
    temp=sent_chain.invoke({"word":word})
    
    # ','으로 끊어지는 문제 해결
    sentence=temp[0]
    for i in range(1,len(temp)):
        sentence+=(', '+temp[i])
    result['sentence'] = sentence
    
    # 문제 만들기
    result['question']=f"위 문장에서 '{word}'가 의미하는 바는 무엇인가요?"
    
    # 보기 만들기
    content=[]
    content.append(meaning)   # 정답 보기 넣기
    
    # 데이터 프레임으로 해당 데이터 가져오기
    # # 랜덤으로 5개를 가져옴
    # random_rows=data.sample(n=5)
    # temp=random_rows['meaning'].tolist()
    # for i in range(len(temp)):
    #     # 보기 4개를 만들었다면
    #     if len(content)==4:
    #         break
    #     # 똑같은 뜻을 가지고 있다면
    #     elif meaning==temp[i]:
    #         continue
    #     # 보기에 추가 할 수 있다면
    #     else:
    #         content.append(temp[i])

    
    # 데이터베이스에서 활용하기
    for i in range(5):
        # 무작위로 뜻 불러오기
        random_word_entry = Word.objects.order_by('?').first()
        data_meaning = random_word_entry.meaning

        if len(content)==4:
            break
        elif meaning==data_meaning:
            continue
        else:
            content.append(data_meaning)
            
    # 보기를 무작위로 섞기
    np.random.shuffle(content)
    
    answers=[]
    for i in range(len(content)):
        # 정답일 경우
        if content[i]==meaning:
            answers.append({"answer":content[i], "correct": True})
        else:
            answers.append({"answer":content[i], "correct": False})

    new_format = {
        "questions": [
            {"Sentence":result['sentence'], "question":result['question'],"answers":answers}
        ]
    }

    return new_format
    
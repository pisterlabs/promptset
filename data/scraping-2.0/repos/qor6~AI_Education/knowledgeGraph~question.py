import openai
import streamlit as st

def query_chat_gpt(question):
    # Chat GPT API에 액세스하기 위한 API 키 설정
    openai.api_key ="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"
    #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"

    # Chat GPT에 질문 보내기
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=question,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.7
    )

    # 가장 적합한 응답 가져오기
    answer = response.choices[0].text.strip()
    return answer

# Chat GPT에 질문 보내기
question = input("질문을 입력하세요: ")
answer = query_chat_gpt(question)
print("답변:", answer)

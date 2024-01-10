import json
from flask import Flask, request, jsonify
import openai
from openai.error import RateLimitError
import os

app = Flask(__name__)

openai.api_key = "YOUR OPENAI KEY"

# GPT-3.5-turbo 모델의 파라미터 설정
model_name = "gpt-3.5-turbo"
temperature = 0.7
max_tokens = 200
top_p = 1
frequency_penalty = 0
presence_penalty = 0

# POST 메서드를 사용해 chat 엔드포인트 생성
@app.route("/chat", methods=['POST'])
def chat():
    # 클라이언트로부터 받은 시스템 입력(system_input)과 사용자 입력(user_input)을 가져옴
    system_input = json.loads(request.json['system'])['system']
    user_input = json.loads(request.json['user'])['user']
    
    # 시스템과 사용자의 대화 내용을 messages 리스트에 담음
    messages = [
        {"role": "system", "content": system_input},
        {"role": "user", "content": user_input}
    ]

    try:
        # OpenAI API를 사용해 GPT-3.5-turbo 모델에게 대화 내용을 전달하고, 모델의 응답을 response_answer에 저장
        response_answer = openai.ChatCompletion.create(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            messages=messages
        )
        # response_answer에서 응답 내용을 content_answer에 저장
        content_answer = response_answer.choices[0].message.content
    except openai.error.OpenAIError as error:
        # OpenAI API 오류 발생 시, 에러 내용을 content_answer에 저장
        content_answer = str(error)

    # 응답 내용과 사용자 입력을 qna 변수에 담아줌
    qna = "user:" + user_input + "\n"+ "ai:" + content_answer
    # 시스템의 대화 내용에 qna 내용을 추가해 messages 리스트에 담음
    messages = [
        {"role": "system", "content": qna + "\n user:와 ai:의 대화를 구분해서 최대한 요약, user의 언어로"}
    ]

    try:
        # OpenAI API를 사용해 GPT-3.5-turbo 모델에게 messages 리스트를 전달하고, 모델의 요약 응답을 response_summary에 저장
        response_summary = openai.ChatCompletion.create(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            messages=messages
        )
        # response_summary에서 요약 내용을 content_summary에 저장
        content_summary = response_summary.choices[0].message.content
    except openai.error.OpenAIError as error:
        # OpenAI API 오류 발생 시, 에러 내용을 content_summary에 저장
        content_summary = str(error)

    return jsonify({'answer': content_answer, 'summary': content_summary})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

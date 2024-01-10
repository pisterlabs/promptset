import os
from openai import OpenAI
import random

# OpenAI API 키 설정
client = OpenAI(api_key='')

class ChatGPTBot:
    def __init__(self):
        self.context = ""

    def generate_response(self, user_input, max_tokens=3000):
        
        bot_initial_messages = [
            "Bot: 안녕하세요! 무엇을 도와드릴까요?",
            "Bot: 안녕하세요, 어떤 정보를 찾고 계신가요?",
            "Bot: 안녕하세요! 오늘 뭐하셨나요?."
            # 추가적인 초기 발화 옵션들을 원하는 만큼 추가.
        ]

        # 봇의 초기 발화를 무작위로 선택
        if not self.context:
            bot_initial_message = random.choice(bot_initial_messages)
            self.context += bot_initial_message
            print(bot_initial_message)
            return


        
        # 사용자 입력을 이전 맥락에 추가
        input_with_context = f"{self.context} User: {user_input}"

        # OpenAI GPT에 쿼리를 보내 응답 생성
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a small talker."},
                {"role": "user", "content": input_with_context},
            ],
            max_tokens=max_tokens,
        )

        # GPT 응답에서 봇의 답변을 추출
        bot_response = response.choices[0].message.content

        # 현재 맥락 업데이트
        self.context += f" User: {user_input}\n Bot: {bot_response}\n"

        return(bot_response)

# 예제 사용
chatgpt_bot = ChatGPTBot()

while True:

    chatgpt_bot.generate_response("")
    user_input = input("사용자: ")
    
    # 종료 조건
    if '종료' in user_input.lower():
        print("대화를 종료합니다.")
        break

    # GPT 봇 응답 생성 및 출력
    bot_response = chatgpt_bot.generate_response(user_input, max_tokens=3000)
    print(bot_response)

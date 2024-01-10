from bardapi import Bard
import openai
import os

#chatGPT에게 채팅 요청 API
def chatGPT(prompt):
    '''
    prompt: HTML TextField에서의 입력(input)
    answer: ChatGPT API를 호출하고, prompt의 결과값을 리스트 형태로 리턴 : [답변]
    '''
    openai.api_key = os.environ.get("AI_CHATGPT_SECRET_KEY")
    
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    print(completion)
    answer = completion.choices[0].message.content
    return [answer]


# Bard에게 채팅 요청 API
def googleBard(prompt):
    '''
    prompt: HTML TextField에서의 입력(input)
    answer: Bard API를 호출하고, prompt의 결과값을 리스트 형태로 리턴 : [답변1, 답변2, 답변3]
    '''
    # token은 구글 바드에다가 아무거나 검색 → 개발자 도구 → Application → __Secure-1PSID의 value 복사
    token = os.environ.get('AI_BARD_SECRET_KEY')
    answer = [Bard(token=token).get_answer(prompt)['choices'][i]['content'][0] for i in range(3)]
    return answer


#두 질문을 합침
def GPT_BARD_answer(prompt):
    '''
     [Bard답변1, Bard답변2, Bard답변3, ChatGPT답변]
    '''
    GPT_ans = chatGPT(prompt)
    Bard_ans = googleBard(prompt)
    return Bard_ans + GPT_ans 


# -----------------------------------------------
# 1. 유저가 질문 입력 
# 2. POST 요청 ({question: "너이름이 뭐니?"}) 
# 3. 서버가 질분데이터 받기 (request.question)
# 4. googleBard함수 호출 (request.question 인자 받기) 
# 5. 함수 실행 및 리턴 
# 6. 프론트로 json 리턴
# -----------------------------------------------

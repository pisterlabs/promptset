import os
import openai
import json

def gpt_response(question, quest_content, user_answer, answer, length):
    updated_question = question.split('600~700자로 글을')[0].strip() + ' 글로 ' + question.split('600~700자로 글을')[1].strip()
    openai.api_key = os.getenv('OPENAI_API_KEY') #호출할 때는 메모장에서 가져오기
    user_content = "문제: " + updated_question +"\n" + '제시문: ' + quest_content + "\n\n" + '사용자 답안: ' + user_answer +"\n"+ '예시 답안' + answer
    message_info = [{
        "role": "system",
        "content": "너는 TOPIK(외국인 및 재외국민을 대상으로 하는 한국어 능력 시험)을 가르치는 선생님이야. 문제와 제시문, 그리고 예시 답안이 주어질거야. 사용자 답안이 문제와 제시문의 내용에 맞게 잘 작성되었는지 채점해줘. 글자 수에 대한 지적은 하지마. 예시 답안은 문제와 제시문에 대한 답변 예시라고 생각해줘. 답안은 JSON 형태로 구성되어야하고 45점이 최고점인 score, Good Points, Weak Point로 구성되어야 해."
        }]
    message_info.append({"role":"user","content":user_content})
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = message_info,
        temperature=0.7,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    answer = response['choices'][0]['message']['content']
    answer = json.loads(answer)
    answer['Length_Check'] = length
    return answer

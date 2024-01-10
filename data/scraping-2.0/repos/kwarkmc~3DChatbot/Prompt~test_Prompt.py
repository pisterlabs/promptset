import os 
import openai 
import sys
import re 
from googletrans import Translator
import datetime

translator = Translator()

openai.api_key = ''
prompt2 = f"""
당신의 임무는 사용자의 답변을 받아 그에 알맞은 답변과 행동을 생성하여 제공하는 것입니다. 
행동은 애니메이션 캐릭터가 움직이는 동작으로 작성해주세요.
또한 행동을 묘사할 때 얼굴이나 표정에 대한 관련 묘사를 제외하고 주로 팔과 다리의 움직임 중심으로 자세하게 묘사하여 작성해주세요. 


행동 예시는 다음과 같습니다. 
user: 안녕!
chatgpt: (양손을 들고 좌우로 흔든다.) 안녕!

user: 오늘 날씨 어때?
chatgpt: (오른손을 이마에 올리고 하늘을 본다.) 오늘 날씨는 좋아요. 

user: 오늘 날씨 어때? 
chatgpt: (런닝머신 위에서 뛰듯이 가벼운 발걸음으로 움직인다.) 오늘은 맑은 날씨라서 실외 활동하기 좋을 것 같아요. 

user: 스쿼시 하고싶다. 
chatgpt: (왼팔을 들어 스쿼시 라켓을 잡는 모션을 보여준다.) 

"""
prompt1 = f"""
당신의 임무는 사용자의 답변을 받아 그에 알맞은 답변과 행동을 생성하여 제공하는 것입니다. 
행동은 얼굴은 제외한 팔과 다리를 움직이는 동작으로 자세하게 묘사하여 작성해주세요. 
또한 행동은 애니메이션 캐릭터가 움직이는 동작으로 작성해주세요

행동 예시는 첫번째는 다음과 같습니다.
사용자: 안녕!
봇: (양손을 들고 좌우로 흔든다.) 안녕!

"""
# 챗봇에 원하는 명령어 작성
prompt = f"""
당신의 임무는 사용자의 답변을 받아 그에 알맞은 답변과 행동을 생성하여 제공하는 것입니다.
행동은 표정을 제외하고 몸을 움직이는 행동으로 작성해주세요. 

행동 예시 첫번째는 다음과 같습니다. 
user: 안녕!
chatgpt: (팔을 머리 위로 들고 흔들며 인사한다) 안녕!

행동 예시 두번째는 다음과 같습니다. 
user: 나 오늘 상 받았어
chatgpt: (앞으로 손뼉을 치며 점프하며 축하한다) 정말 축하해! 

행동 예시 세번째는 다음과 같습니다. 
user: 스쿼시 하고싶다. 
chatgpt: (왼팔을 들어 스쿼시 라켓을 잡는 모션을 보여준다.)

행동은 몸을 움직이는 동작에 중점을 두고 작성해주세요.
움직이는 동작을 정확하게 묘사하여 작성해주세요.
예를 들어 머리 위로 손을 흔들고 싶다면 '손을 머리 위로 올려 흔든다'와 같은 문장을 작성해주세요.
행동은 괄호 안에 작성하고 사용자 답변에 알맞은 답변과 함께 답해주세요.
"""

messages = [{'role':'system', 'content':prompt2}]

# 반복적인 대화 
while True:
    message = input('user: ')
    if message:
        messages.append(
            {'role':'user', 'content':message},
        )
        chat = openai.ChatCompletion.create(
            model = 'gpt-3.5-turbo', 
            presence_penalty = 0.7,
            frequency_penalty = 0.7,
            max_tokens = 150,
            messages = messages
        )
    reply = chat.choices[0].message.content
    print(f'chatgpt: {reply}')

    # 답변에서 () 부분만 출력
    p = re.compile('\(([^)]+)')
    action = p.findall(reply)
    action = " ".join(action)
    action_trans = translator.translate(action, dest='en', src='ko')
    action_trans = action_trans.text
    print('행동: ', action)
    print('Action: ', action_trans)

    current_time = datetime.datetime.now()
    file_name = current_time.strftime("%Y-%m-%d_%H-%M-%S") + ".txt"

    file_path = './motion-diffusion-model/TEXT/' + file_name

    #해당 경로에 영어로 번역된 행동 1.txt 파일로 저장
    with open(file_path, 'w') as f:
        f.write(action_trans)


    messages.append({'role':'assistant', 'content':reply})


'''


    return response.choices[0].message.content




response = get_completion(prompt)
print(response)

'''
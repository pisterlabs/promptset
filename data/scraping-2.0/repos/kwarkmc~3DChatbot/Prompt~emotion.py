import openai 
import sys 
import re 
openai.api_key = ''

prompt = f"""
당신의 임무는 사용자의 답변을 받아 그에 알맞은 답변과 행동을 생성하여 제공하는 것입니다. 
행동은 애니메이션 캐릭터가 움직이는 동작으로 작성해주세요.
또한 행동을 묘사할 때 얼굴이나 표정에 대한 관련 묘사를 제외하고 주로 팔과 다리의 움직임 중심으로 자세하게 묘사하여 작성해주세요. 

행동 예시는 다음과 같습니다. 
user: 안녕!
chatgpt: (팔을 머리 위로 들고 흔들며 인사한다) 안녕! 

user: 오늘 날씨 어때?
chatgpt: (오른손을 이마에 올리고 하늘을 본다.) 오늘 날씨는 좋아요. 

user: 오늘 날씨 어때? 
chatgpt: (런닝머신 위에서 뛰듯이 가벼운 발걸음으로 움직인다.) 오늘은 맑은 날씨라서 실외 활동하기 좋을 것 같아요. 

user: 스쿼시 하고싶다. 
chatgpt: (왼팔을 들어 스쿼시 라켓을 잡는 모션을 보여준다.) 

그리고 해당 ()속에 있는 행동에서 신체의 움직임을 제외한 얼굴과 표정 묘사가 포함되어 있다면 다음 예시처럼 변환하여 작성해주세요. 
예시는 다음과 같습니다. 
chatgpt: (격한 표정으로 이마에 손을 얹고 씁쓸하게 미소를 짓는다)

변환된 예시
chatgpt: (이마에 손을 얹는다)

"""

# 감정 분류 위한 프롬프트
emotion_pro = f"""
당신의 임무는 chatgpt의 답변을 받아 답변에서 표현하고 있는 감정이 무엇인지 작성하는 것입니다. 

감정 유형은 ['슬픔', '행복', '두려움', '놀람', '기쁨', '분노', '중립'] 총 7개입니다. 
위 7개의 감정 유형 중에서 하나를 선택해 작성해주세요.
"""
messages = [{'role':'system', 'content':prompt}]

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

    emotion = openai.ChatCompletion.create(
        model = 'gpt-3.5-turbo',
        temperature = 0.5,
        messages = [{'role':'system', 'content':emotion_pro},
                    {'role':'user', 'content':reply}]
    )
    emo = emotion.choices[0].message.content 
    print(f'chatgpt: {reply}')
    print(f'emotion: {emo}')

    # 답변에서 () 부분만 출력
    #p = re.compile('\(([^)]+)')
    #action = p.findall(reply)
    #print('행동: ', action[0])

    messages.append({'role':'assistant', 'content':reply})


'''


    return response.choices[0].message.content




response = get_completion(prompt)
print(response)

'''
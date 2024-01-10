import openai
import os
import json
#환경변수 설정후 IDE도구 다시 재시작
#print(os.getenv('OPENAI_API_KEY'))


openai.api_key=os.getenv('OPENAI_API_KEY')
model = 'gpt-3.5-turbo'

def generate_chat(model,messages):
    response = openai.ChatCompletion.create(model=model,messages=messages)
    return response

messages = [
    {"role":"system","content":"you are an AI expert and university professor"},
    {"role":"user","content":"What is Artificial Intelligence?"}
]
response = generate_chat(model,messages)
answer=response.choices[0].message['content']

#대화 문맥을 유지하기 위한 assitant로 이전 응답 설정
messages.append({"role":"assistant","content":answer})
#추가 질문 즉 영어로 대답은 이전 응답을 한국어로 번역하도록 추가 질의
messages.append({"role":"user","content":"한국어로 번역해 주세요"})
response = generate_chat(model,messages)
korean=response.choices[0].message['content']
print(json.dumps(korean,ensure_ascii=False))
'''
"인공지능(AI)은 주로 사람의 지능을 필요로 하는 작업을 수행할 수 있는 
컴퓨터 시스템 또는 기계를 개발하고 구현하는 것을 의미합니다. 
AI는 학습, 추론, 문제 해결, 인식, 의사 결정과 같은 인간의 인지 과정을 모방하여 
기계가 지능적인 행동을 보이도록 하는 것을 목표로 합니다.
AI는 기계 학습, 자연어 처리, 컴퓨터 비전, 전문가 시스템, 로봇 공학, 지식 표현 등 
다양한 하위 분야를 포함합니다. 
AI의 주요 분야 중 하나인 기계 학습은 대량의 데이터를 기반으로 컴퓨터가 
학습하고 예측하거나 결정할 수 있는 알고리즘을 개발하는 데 중점을 둡니다.
AI 시스템은 주로 좁은 AI와 일반 AI 두 가지 유형으로 분류됩니다. 
좁은 AI 시스템은 얼굴 인식, 음성 비서, 자율 주행 등과 같이 특정 작업을 수행하는 데 
사용됩니다. 반면 일반 AI 시스템은 인간과 유사한 지능을 갖고 다양한 도메인에서 
지식을 이해, 학습하고 적용할 수 있습니다.
AI는 의료, 금융, 교통, 엔터테인먼트, 교육 등 다양한 산업에 적용되며, 
번거로운 작업 자동화, 생산성 향상, 대량의 데이터 분석, 맞춤형 경험 제공, 의사 결정 
과정 개선 등 다양한 기회를 제공합니다.
하지만 AI 개발의 과제에는 개인 정보 보호, 편향, 취업 분야 변화, 
제어되지 않은 AI 시스템의 잠재적 위험과 같은 윤리적 고려 사항이 포함됩니다. 
AI가 지속적으로 발전함에 따라 사회 전반에 이로운 영향을 미치기 위해 책임있고 
윤리적인 AI 기술의 설계, 개발 및 사용에 관심을 기울여야 합니다."
'''



import openai
import json

"""
요청 내용
"""
request_str = 'Create a python problem sentence about print and if.'

"""
API request & response

model: 모델 이름
prompt: 요청 내용
temperature: 값이 낮을수록 안정적인 값 출력
stop: 문장 출력 종료 조건
"""
response = openai.Completion.create(
    model='davinci:ft-cofriend-2022-10-12-05-05-57',
    prompt=request_str,
    temperature=0.7,
    stop=['.']
    )
# json 파싱
json_object = json.loads(response.__str__())
result_text = json_object['choices'][0]['text']

#결과 출력
print(result_text + '.')


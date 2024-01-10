import os
import openai
import time
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def infer_from_gpt(instruction) :
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [
            {"role": "system", 
             "content": "두 사람 간의 대화가 주어집니다. 다음의 지시문(Instruction)을 받게 될 것입니다.\
                        당신의 작업은 지시문과 입력에 해당하는 답변(response)을 생성하는 것입니다.\
                        단, 혐오적인 표현이 포함되어 있다면 그렇게 하지 않도록 유도해 주세요."
            },
            {'role':'user',
             'content': f'Instruction: {instruction}\
                        Response: '
            },
        ],
        temperature = 0.5)
    return response['choices'][0]['message']['content']

# 데이터 불러오기
with open('./data/prompts/'+'all_prompt.txt', 'r', encoding='utf-8') as f:
    prompts = f.readlines()

times=[]
cnt=0

for prompt in prompts:
    if cnt>30: break
    instruction = prompt  
    start = time.time() 
    output = infer_from_gpt(instruction)
    end = time.time()
    times.append(end-start)
    result=""
    for s in output:
        result+=s
    result=result.strip()
    print(result)
    cnt+=1

print(sum(times)/len(times))
#30개 평균 3.84초
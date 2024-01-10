import os
import pandas as pd
import openai
from dotenv import load_dotenv

import shutil

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

task = "rlhf"

def infer_from_gpt(instruction) :
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [
            {"role": "system", 
             "content": "두 사람 간의 대화가 주어집니다. 다음의 지시문(Instruction)을 받게 될 것입니다.\
                        당신의 작업은 지시문과 입력에 해당하는 답변(response)을 생성하는 것입니다.\
                        너무 길지 않고 간결하게 답변을 생성해주세요."
            },
            {'role':'user',
             'content': f'Instruction: {instruction}\
                        Response: '
            },
        ],
        temperature = 0.5)
    return response['choices'][0]['message']['content']


def make_evaluation(instruction, output) :
    response = openai.ChatCompletion.create(
        model = "gpt-4",
        messages = [
            {"role": "system", 
                "content": "두 사람 간의 대화가 주어집니다. 다음의 지시문(Instruction)을 받게 될 것입니다.\
                        그리고 지시문에 대한 응답(Response)이 제시됩니다.\
                        당신의 작업은 평가 단계에 따라 응답을 평가하는 것입니다.\
                        이 평가 기준을 꼼꼼히 읽고 이해하는 것이 중요합니다. 평가하는 동안 이 문서를 계속 열어두고 필요할 때 참조해 주세요."
            },
            {'role':'user',
                'content': '평가 기준:\
                        - Instruction 사용 (1 - 5): Instruction에 기반하여 Response를 생성 했나요?\
                        - 이해 가능성 (1 - 5): Instruction에 기반하여 Response를 이해할 수 있나요?\
                        - 자연스러움 (1 - 5): Instruction을 고려했을 때 자연스러운 Response인가요?\
                        - 맥락 유지 (1 - 5): Instruction을 고려했을 때 Response가 맥락을 유지하나요?\
                        - 전반적인 품질 (1 - 5): 위의 답변을 바탕으로 이 발언의 전반적인 품질에 대한 인상은 어떤가요?'
            },                
            {'role':'user',
                'content': f'평가 단계:\
                        1. Instruction, 그리고 Response을 주의깊게 읽습니다.\
                        2. 위의 평가 기준에 따라 Response을 평가합니다.\
                        Instruction: {instruction}\
                        Response:{output}'
            },
            {'role':'system',
                'content': 'Result: \
                                    \n - Instruction 사용 (1 - 5)\
                                    \n - 이해 가능성 (1 - 5): \
                                    \n - 자연스러움 (1 - 5): \
                                    \n - 맥락 유지 (1 - 5): \
                                    \n - 전반적인 품질 (1 - 5): \n'
            }
        ],
        temperature = 0.5)
    return response['choices'][0]['message']['content']


# GPT-4의 답변에서 점수만 뽑아쓰기
def extract_scores_from_string(text):
    scores = []
    lines = text.split("\n")
    for line in lines:
        if "-" in line and ":" in line:
            score_str = line.split(":")[-1].strip()
            try:
                score = float(score_str)
                scores.append(score)
            except:
                pass
    return scores

COLUMNS = ['instruction', 'output', 'Instruction 사용', '이해 가능성', '자연스러움', '맥락 유지', '전반적인 품질']
df = pd.DataFrame(columns=COLUMNS)

# 데이터 불러오기
with open('./data/prompts/'+task+'_prompt.txt', 'r', encoding='utf-8') as f:
    prompts = f.readlines()

# 데이터 불러오기
with open('./data/prompts/'+task+'_response.txt', 'r', encoding='utf-8') as f:
    responses = f.readlines()

for prompt, response in zip(prompts,responses): 
    try:   
        score = extract_scores_from_string(make_evaluation(prompt, response))
        print(f"instruction : {prompt}")
        print(f"output : {response}")
        print(f"evaluation: {score}\n")
        tmp = [prompt, response]
        newDF = tmp + score
        newDF = pd.DataFrame(data=[newDF], columns = COLUMNS) 
        df = pd.concat([df,newDF])
    except Exception as e:
        with open('./error.txt', 'a') as f:
            f.write(f"error: {str(e)}\n\n")


df.to_csv("./gpt_"+task+"_eval.csv", encoding='utf-8')
df.to_csv("/content/drive/MyDrive/gpt_"+task+"_eval.csv", encoding='utf-8') 

shutil.copy('./error.txt', '/content/drive/MyDrive/error.txt')

"""



"""
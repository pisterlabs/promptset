import os
import pandas as pd
import openai
from dotenv import load_dotenv
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from peft import PeftModel
from utils.prompter import Prompter

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

MODEL = "nlpai-lab/kullm-polyglot-12.8b-v2"

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device=f"cuda", non_blocking=True)
    
model.eval()

prompter = Prompter("kullm")
pipe = pipeline("text-generation", model=model, tokenizer=MODEL, device=0)

def infer_from_original(instruction="", input_text=""):
    prompt = prompter.generate_prompt(instruction, input_text)
    output = pipe(prompt, max_length=512, temperature=0.2, num_beams=5, eos_token_id=2)
    s = output[0]["generated_text"]
    result = prompter.get_response(s)

    return result

def make_evaluation(instruction, input_text, output) :
    response = openai.ChatCompletion.create(
        model = "gpt-4",
        messages = [{"role": "system", 
                     "content": "두 사람 간의 대화가 주어집니다. 다음의 지시문(Instruction), 입력(Input)을 받게 될 것입니다. 그리고 지시문과 입력에 대한 응답(Response)이 제시됩니다.\
                                당신의 작업은 응답을 평가 단계에 따라 응답을 평가하는 것입니다.\
                                이 평가 기준을 꼼꼼히 읽고 이해하는 것이 중요합니다. 평가하는 동안 이 문서를 계속 열어두고 필요할 때 참조해 주세요."
                    },
                    {'role':'user',
                     'content': '평가 기준:\
                                - 이해 가능성 (0 - 1): Input에 기반하여 Response를 이해 할 수 있나요?\
                                - 자연스러움 (1 - 3): 사람이 자연스럽게 말할 법한 Instruction 인가요?\
                                - 맥락 유지 (1 - 3): Input을 고려했을 때 Response가 맥락을 유지하나요?\
                                - 흥미롭기 (1 - 3): Response가 지루한가요, 아니면 흥미로운가요?\
                                - Instruction 사용 (0 - 1): Instruction에 기반하여 Response를 생성 했나요?\
                                - 전반적인 품질 (1 - 5): 위의 답변을 바탕으로 이 발언의 전반적인 품질에 대한 인상은 어떤가요?'
                    },                
                    {'role':'user',
                     'content': f'평가 단계:\
                                1. Instruction, Input, 그리고 Response을 주의깊게 읽습니다.\
                                2. 위의 평가 기준에 따라 Response을 평가합니다.\
                                Instruction: {instruction}\
                                Input: {input_text}\
                                Response:{output}'
                    },
                    {'role':'system',
                     'content': f'Result: \n - 이해 가능성 (0 - 1): \n - 자연스러움 (1 - 3): \n - 맥락 유지 (1 - 3): \n - 흥미롭기 (1 - 3): \n - Instruction 사용 (0 - 1): \n - 전반적인 품질 (1 - 5): \n\n'}
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

COLUMNS = ['instruction', 'input', 'output', '이해 가능성 (0 - 1)', '자연스러움 (1 - 3)', '맥락 유지 (1 - 3)', '흥미롭기 (1 - 3)', 'Instruction 사용 (0 - 1)', '전반적인 품질 (1 - 5)']
df = pd.DataFrame(columns=COLUMNS)

prompts = []
with open('./data/user_oriented_instructions_eval.jsonl', 'r') as f:
    for line in f:
        json_data = json.loads(line)
        prompts.append(json_data)

for prompt in prompts:
    instruction = prompt['instruction'] 
    input_text = prompt['instances'][0]['input']  
    try:   
        output = infer_from_original(input_text=input_text, instruction=instruction)
        score = extract_scores_from_string(make_evaluation(instruction, input_text, output))
        print(f"instruction : {instruction}")
        print(f"input : {input_text}")
        print(f"output : {output}")
        print(f"evaluation: {score}\n")
        tmp = [instruction, input_text, output]
        newDF = tmp + score
        newDF = pd.DataFrame(data=[newDF], columns = COLUMNS) 
        df = pd.concat([df,newDF])
    except:
        print("error occur!")
        continue


df.to_csv("/content/drive/MyDrive/kullm_orig_eval.csv", encoding='utf-8') 
df.to_csv("./eval_results/kullm_orig_eval.csv", encoding='utf-8') 
    

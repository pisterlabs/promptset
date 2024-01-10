import os
import sys
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LLaMAForCausalLM, LLaMATokenizer
import json
import time
import re

'''
device = "cuda:0"


model = LLaMAForCausalLM.from_pretrained(
    "alpaca-lora/13B_HF/",
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(
    model,
    "alpaca-lora/lora-alpaca-13B/",
    torch_dtype=torch.float16,
)
tokenizer = LLaMATokenizer.from_pretrained("alpaca-lora/13B_HF/tokenizer.model")

# unwind broken decapoda-research config
model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2

def remove_common_prefix(i_str2):
        r_str2=[]
        breakpoint()
        for i in range(len(i_str2)):
            # str1=i_str1[i]
            str2=i_str2[i]
            pattern = re.compile(re.escape("### Ins"))
            match = pattern.search(str2)
            if(match):
                str2 = str2[:match.start()]
            r_str2.append(str2)
        return r_str2


model.eval()
if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

'''

# We mix 3 datasets - MATH, StratergyQA, AQuA to produce 5k training examples
# MATH counting and probability - 1k
# MATH intermediate algebra - 500
# MATH number_theory - 500
# StrategyQA - 1k
# AQuA - 2.5k

def create_dataset_StrategyQA(filename, bs):
    # open dev.json file. this file has data in the format [{"question": "question", "answer": "answer", "facts":facts, "decomopsition": decomposition}]. make arrays of questions, answers, facts and decompositions
    Questions = []
    Answers=[]
    Facts = []
    Decomposition = []
    with open(filename) as f:
        data = json.load(f)

    for item in data:
        question = item['question']
        answer = item['answer']
        facts = item['facts']
        fact_str = ""
        for fact in facts:
            fact_str += fact + "\n"
        decomposition = item['decomposition']
        dec = ""
        for decomp in decomposition:
            dec += decomp + '\n'

        Questions.append(question)
        Answers.append(answer)
        Facts.append(fact_str)
        Decomposition.append(dec)
    
    return Questions, Answers, Facts, Decomposition

def create_dataset_MATH(directory, bs):
    Questions = []
    Reasoning = []

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r") as file:
                data = json.load(file)
                
                # Extract problem and solution from the JSON data
                Questions.append(data["problem"])
                rs = data["solution"].split('.')
                step = ""
                for r in rs:
                    step+=r+'\n'
                Reasoning.append(step[:-2])


    return Questions, Reasoning

def create_dataset_AQuA(filename, bs):
    Questions = []
    Answers=[]
    Rationale = []
    with open(filename, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        question = result['question']
        rs = result["rationale"]
        step = ""
        for r in rs:
            step+=r+'\n'
        Rationale.append(rs)
        # Answers.append(step)
        options = result['options']
        for option in options:
            question += " " + option + " "
        # Questions.append(result['question'])
        # Questions.append(result['problem'])
        Questions.append(question)
        # Answers.append(extract_answer(result['answer']))
        # Answers.append(result['correct'])
    return Questions, Rationale

Q_stratergy, A_stratergy, COT_stratergy, SQ_stratergy = create_dataset_StrategyQA("./alpaca-lora/strategyqa/train.json", 1)
# Q_pnc, R_pnc = create_dataset_MATH("./alpaca-lora/MATH/train/counting_and_probability", 1)
# Q_NT, R_NT = create_dataset_MATH("./alpaca-lora/MATH/train/number_theory", 1)
# Q_ialg, R_ialg = create_dataset_MATH("./alpaca-lora/MATH/train/intermediate_algebra", 1)
# Q_AQuA, R_AQuA = create_dataset_AQuA("./alpaca-lora/AQuA/train.json", 1)

# Q = Q_pnc[:1000] + Q_NT[:500] + Q_ialg[:500] + Q_AQuA[:3000]
# R = R_pnc[:1000] + R_NT[:500] + R_ialg[:500] + R_AQuA[:3000]
breakpoint()
for i in range(len(Q_stratergy)):
    with open('./alpaca-lora/Train_Data_Part2_Strategy.json', 'a') as f:
            json.dump({"question": Q_stratergy[i], "Reasoning": COT_stratergy[i], "sub-questions": SQ_stratergy[i]}, f)
            f.write(os.linesep)
        
sys.exit()
Q = Q_AQuA[337:3000]
R = R_AQuA[337:3000]
breakpoint()
pr = '''
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction: Create subquestions for the question - "If a person walks at 16 km/hr instead of 10 km/hr, he would have walked 20 km more. The actual distance traveled by him is: A)50 km  B)56 km  C)60 km  D)70 km  E)33.3 km ", based on the steps given in the input
### Input: 
Let the actual distance travelled be x km.
x/10 = (x+20)/16
16x = 10x + 200
6x = 200
x = 33.3 km.
answer :E
### Response: subquestions - 
What is the time taken if he would have travelled at 10km/hr?
What is the time taken to travel 20 km more at a speed of 16km/hr?
What is the equation to be solved?
'''

prompts = []

for i in range(len(Q)):
      p = ""
      p += f'### Instruction: Create subquestions for the question- \"' + Q[i] + '\", based on the steps given in the input' + '### Input: ' + R[i] + '\n' + '### Response: subquestions - '
      prompts.append(pr+p)

'''
generation_config = GenerationConfig(
    temperature=0.95,
    top_p = 0.18
)
with open('./alpaca-lora/PnC.json', 'a') as f:
    for i in range(len(prompts)-1):
        input_ids = tokenizer(prompts[i+1], return_tensors="pt", padding=True, truncation = True, max_length = 2048).input_ids.to(device)
        outputs=model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=128,
                )
        result = []
        breakpoint()
        for s in outputs.sequences:
                result.append(remove_common_prefix(tokenizer.decode(s[input_ids.shape[-1] :])))
                json.dump({"question": questions[i+1], "answer": reasoning_steps[i+1], "sub-questions": result[0]}, f)
                f.write(os.linesep)

        print(result[0])
'''


import os
import openai

# Load your API key from an environment variable or secret management service
# openai.api_key = "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"
openai.api_key = "060db6b6c3ff468ca2215e0ef75b9cc1"
openai.api_base = "https://kglm.openai.azure.com/" # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
openai.api_type = 'azure'
openai.api_version = '2023-05-15' # this may change in the future
deployment_name='kglm-text-davinci-003'

for i in range(len(prompts)):
    try:
        response = openai.Completion.create(engine=deployment_name, prompt= prompts[i], temperature=0, max_tokens=512)
        # Open a json file, add question from the question array, answer from the answer array and response recieved to it ans save it
        with open('./alpaca-lora/Train_Data_Part2_AQuA.json', 'a') as f:
            json.dump({"question": Q[i], "Reasoning": R[i], "sub-questions": response["choices"][0]["text"]}, f)
            f.write(os.linesep)
        
        # print(response["choices"][0]["text"])
        time.sleep(0.25)
        # breakpoint()
    except:
        print("Error occured at question: ", Q[i])

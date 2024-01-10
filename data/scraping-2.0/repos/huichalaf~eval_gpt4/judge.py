import openai
import os, sys
import json
import random
import tiktoken
from functions import call_gpt_4_eval, call_model
from time import time
import dotenv
from functions import call_gpt_4_judge, call_model
import multiprocessing

def call_with_timeout(func, args, timeout=60):
    with multiprocessing.Pool(1) as p:
        result = p.apply_async(func, args)
        try:
            return result.get(timeout=timeout)
        except multiprocessing.TimeoutError:
            print("Function call timed out")
            return None

dotenv.load_dotenv()
with open(sys.argv[2]) as f:
    config = json.load(f)

start = time()
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

file_questions = sys.argv[1]

openai.api_key = os.getenv("OPENAI_API_KEY")

goal = config["goal"]
system_message_chats = config["system_message"]
questions = []
with open(file_questions) as f:
    texto = f.readlines()
    for line in texto:
        questions.append(line)

print(len(questions))
#las revolvemos
random.shuffle(questions)

model_judge = config["judge_model"]
model_base = config["base_model"]
model_tested = config["tested_model"]
resultados = []
#questions = questions[:20]
for question in questions:
    index = questions.index(question)
    print(round(index/len(questions)*100),"%")
    question = question.replace("\n", "")
    response = ''
    question += ". Let's think step by step."
    try:
        response1 = call_with_timeout(call_model, [model_base, system_message_chats, question])
    except Exception as e:
        print(e)
        continue

    try:
        response2 = call_with_timeout(call_model, [model_tested, system_message_chats, question])
    except Exception as e:
        print(e)
        continue

    try:
        response = call_with_timeout(call_gpt_4_judge, [goal, model_judge, question, response1, response2])
        resultados.append(response)
        print(response)
    except Exception as e:
        print(e)
        continue
print(resultados)
res = []
for i in resultados:
    try:
        res.append(int(i))
    except:
        pass
print(sum(res)/len(res))
#contamos cuantas veces gano cada modelo
print("base: ", resultados.count("1"))
print("tested: ", resultados.count("2"))
print(time()-start)
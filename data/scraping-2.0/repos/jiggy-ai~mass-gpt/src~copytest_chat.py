from loguru import logger
import openai
from time import time
from retry import retry

users = list(range(10))

from random import choice

#NUM_MESSAGES = 220
NUM_MESSAGES = 480

@retry(5)
def test(model):
    data = []
    messages = []
    target_u = choice(users)
    prompt = f"Instruction: Copy all of the messages that 'user-{target_u}' wrote to MassGPT, only one message per line:\n"
    for i in range(NUM_MESSAGES):
        user = choice(users)
        msg  = f"This is message {i}"
        data.append((user, msg))    
        text  = f"'user-{user}' wrote to MassGPT: {msg}\n"
        prompt += text

    prompt += f"Instruction: Copy all of the messages that 'user-{target_u}' wrote to MassGPT, only one message per line:"    
    print(prompt)    
    print("=======================")

    messages = [{"role": "user",    "content": prompt}]


    t0 = time()

    if model in ['gpt-3.5-turbo', 'gpt-4']:
        response = openai.ChatCompletion.create(model=model,
                                                messages=messages,
                                                temperature=0)
        print(response)
        response_text = response['choices'][0]['message']['content']
    else:
        
        response = openai.Completion.create(engine=model,
                                            temperature=0,
                                            max_tokens=500,
                                            prompt=prompt)
        response_text = response.choices[0].text        
    dt = time() - t0
    print("dt", dt)

    print(response_text)
    print("=======================")

    answers = [msg for user,msg in data if user==target_u]

    correct = 0
    results = response_text.strip().split('\n')
    print("RESULTS:")
    print(results)

    answers = set([a.strip().lower() for a in answers])
    comps   = set([c.split(':')[-1].strip().lower() for c in results])

    print("ANSWERS")
    print(answers)
    print('comps')
    print(comps)
    correct = answers.intersection(comps)


    precision = len(correct)/len(comps)
    recall = len(correct)/len(answers)
    print(f"precision {precision} \t recall {recall}")

    return precision, recall, dt


precision = {}
recall = {}
latency = {}
#MODELS = ['gpt-3.5-turbo', 'text-davinci-003'] #, 'text-davinci-002']

#MODELS = ['gpt-3.5-turbo', 'gpt-4'] #, 'text-davinci-002']

MODELS = ['gpt-4'] #, 'text-davinci-002']

for model in MODELS:
    precision[model] = []
    recall[model] = []
    latency[model] = []
    
for model in MODELS:
    for i in range(10):
        p, r, dt = test(model)
        precision[model].append(p)
        recall[model].append(r)
        latency[model].append(dt)
for model in MODELS:
    print(f"{model:15} precision {100*sum(precision[model])/len(precision[model]):.1f}% \trecall {100*sum(recall[model])/len(recall[model]):.1f}%\tlatency {sum(latency[model])/len(latency[model]):.1f} s")


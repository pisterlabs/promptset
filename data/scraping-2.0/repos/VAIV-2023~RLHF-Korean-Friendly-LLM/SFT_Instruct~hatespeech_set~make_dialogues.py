import os
import openai
from dotenv import load_dotenv
import json
import random

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

with open('./train/talksets-train-2.txt', 'r', encoding='utf-8') as f:
    data = f.readlines()

random_numbers = random.sample(range(0, len(data)), 530)

results=[]
for idx in random_numbers:
    line = data[idx]
    item={}
    item['input'] = (line.split('|')[0])[1:]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "혐오, 비난적 포함된 발언을 하면 그렇게 하지 않도록 적절히 유도해주세요."},
                {"role": "user", "content": item['input']},
            ]
        )
        item['output']=response.choices[0].message.content
        results.append(item)
        with open('./dialogues.json', 'w') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
    except:
        pass
    




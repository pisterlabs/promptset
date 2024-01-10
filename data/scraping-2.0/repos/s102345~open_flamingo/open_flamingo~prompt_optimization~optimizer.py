from dotenv import load_dotenv
import openai
import os, json, re
from appdata import root

class Optimizer():
    def __init__(self):
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.init()
        
    def init(self):
        print("Optimizer initialized!")
        if not os.path.exists(f'{root}/tmp'):
            os.mkdir(f'{root}/tmp')
        json.dump([], open(f'{root}/tmp/solutions.json', 'w'))

    def generate(self, meta_prompt):
        messages = [
            {"role": "system", "content": meta_prompt},
        ]
        past_solution = json.load(open(f'{root}/tmp/solutions.json', 'r'))

        for solution in past_solution:
            messages.append({"role": "assistant", "content": solution['solution']})
            
        completion  = openai.ChatCompletion.create(
            model='gpt-4',
            messages=messages
        )

        tmp = re.findall(r'\[.*?\]', completion.choices[0].message['content'])
        # Not in [] format
        if len(tmp) == 0:
            new_solution = completion.choices[0].message['content']
        else:
            new_solution = tmp[0][1: -1]

        past_solution.append({'solution': new_solution})
        json.dump(past_solution, open(f'{root}/tmp/solutions.json', 'w'), indent=4)

        return new_solution

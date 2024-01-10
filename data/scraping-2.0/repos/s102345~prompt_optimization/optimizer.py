from dotenv import load_dotenv
import openai
from tenacity import wait_random_exponential, stop_after_attempt, retry
import os, json, re
from appdata import root

class Optimizer():
    def __init__(self):
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.init()
        
    def init(self):
        if not os.path.exists(f'{root}/tmp'):
            os.mkdir(f'{root}/tmp')
        json.dump([], open(f'{root}/tmp/solutions.json', 'w'))
        self.messages = []
        print("Optimizer initialized!")

    @retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(10))
    def call_API(self):
        completion  = openai.ChatCompletion.create(
            model='gpt-4',
            messages=self.messages
        )
        return completion

    def prepare_messages(self, meta_prompt):
        self.messages = [
            {"role": "system", "content": meta_prompt},
        ]

        past_solution = json.load(open(f'{root}/tmp/solutions.json', 'r'))

        for solution in past_solution:
            self.messages.append({"role": "assistant", "content": solution['solution']})

    def generate(self, meta_prompt):
        print("Generating solution...")
        if self.messages == []:
            self.prepare_messages(meta_prompt)

        past_solution = json.load(open(f'{root}/tmp/solutions.json', 'r'))

        completion = self.call_API()        

        tmp = re.findall(r'\[.*?\]', completion.choices[0].message['content'])
        # Not in [] format
        if len(tmp) == 0:
            new_solution = completion.choices[0].message['content']
        else:
            new_solution = tmp[0][1: -1]

        past_solution.append({'solution': new_solution})
        json.dump(past_solution, open(f'{root}/tmp/solutions.json', 'w'), indent=4)

        print("Generating solution done!")
        return new_solution

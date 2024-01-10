import openai
from utils import build_prompt

class EntityStateGeneration(): 
    '''function to generate entity state changes given the goal, context, and current step of a procedure. '''
    def __init__(self):
        pass
    
    def gpt4(self, prompt):
        res = openai.Completion.create(
                    engine='text-davinci-004',
                    prompt=prompt,
                    temperature=0.7,
                    max_tokens=300,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=['\n\n']
                )
        return res
    
    def forward(self, goal, context, cur_step):
        prompt = build_prompt(goal, context, cur_step)
        answer = self.gpt4(prompt)
        return answer

entity_model = EntityStateGeneration()


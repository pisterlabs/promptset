import sys
import os
import openai
from config import GPT3_API_KEY

openai.api_key = GPT3_API_KEY
openai.Model.list()

pre_prompt = '''
I am about to provide a question in natural language and I want you to isolate the variables and their values. 
If it is not known, just say variable=?? or provide the formula to calculate it. 
Include any formulas inside the prompt in your answer that could not reasonably be known without the prompt. 
Do not inlclude "known" formulas (ie y=mx+b).
Provide this is plain csv format with a start and stop token. 
Explicitly state which variable is being asked to solve for.
Be as explicit as possible, do not abbreviate any variable name.
Only provide the csv string, no need for anything else.
example: "<start>desired_variable=variable_three;variable_one=5;variable_two=7;variable_three=??<stop>". 
ok, here is the question: 
'''

question_0 = '"Albert has 10 apples. Laura has 4 apples. Albert gives Laura 5 apples. how many more apples does albert have than laura?"'
question_1 = '"A block of mass 5.0 kg is pushed across a horizontal surface by a horizontal force of 20 N. If the frictional force between the block and the surface is 8.0 N, what is the acceleration of the block?"'
question_2 = '"Solve for x: x + y = 5;y + x = 6"'

s_prompt = pre_prompt + question_0

def get_response(s_model,message):
    
    response = openai.ChatCompletion.create(
        model = 'gpt-4-0314',
        temperature = 1,
        messages = [
            {"role": "user", "content": message}
        ]
    )
    print(response.__dict__)
    return response.choices[0]["message"]["content"]

# define the model, max tokens, and temperature
s_model="gpt-4-0314"
n_max_tokens=250
n_temperature = 0.9

# generate the answer
answer = get_response(s_model,s_prompt)
#remove \n from answer
answer = answer.replace("\n", "")

# print the completion
print(f'Prompt: {s_prompt}')
print('')
print(f'{s_model} Answer: {answer}')
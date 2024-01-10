from utils import *
from prompts.declarative_eight_shot import DECLARATIVE_EIGHT_SHOT
import openai
import time

#results - three-shot - 176/222 correct - 79.279% accuracy
#results - eight-shot - 158/222 correct - 71.171% accuracy

st = time.time()

def get_file_contents(filename):
    try:
        with open(filename, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        print("'%s' file not found" % filename)

api_key = get_file_contents('api_key.txt')
openai.api_key = api_key

with open('declarative-math-word-problem/algebra222.csv') as f:
    questions = [i.split(',')[0] for i in f.readlines()]

with open('declarative-math-word-problem/algebra222.csv') as f:
    answers = [i.split(',')[1] for i in f.readlines()]

solver_answers = []

for i in range(0, 24):
    eq_list = get_declarative_equations(model='text-davinci-003', question=questions[i], prompt=DECLARATIVE_EIGHT_SHOT, max_tokens=600, stop_token='\n\n\n', temperature=0)
    answer = get_final_using_sympy(eq_list)
    solver_answers.append(answer)

print(*solver_answers, sep = '\n')

et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')
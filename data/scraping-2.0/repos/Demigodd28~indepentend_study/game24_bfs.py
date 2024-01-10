import dotenv
import os
from openai import OpenAI
from game24 import *
import pandas as pd
import re
import json
import sympy

class Node:
    def __init__(self, answer, value):
        self.answer = answer
        self.value = value
        self.parent_node = None
        self.children_node = None

    def append_parent(self, parent):
        self.parent_node = parent
    
    def append_children(self, children):
        self.children_node = children

def Load_data():
    data = pd.read_csv('24.csv')
    return data['Puzzles']

def Parse_propose_response(response: str):
    answers = list(response.strip().split('\n'))
    return answers

def Parse_value_response(response):
    answer = response.split('\n')[-1]
    return answer

def Generator(llm: OpenAI, model, temperature, nodes, k, id):
    new_nodes = list()
    print('Generator: ')
    with open('record.txt', 'a') as file:
        file.write('\nTop_b\n')
        for node in nodes:
            file.write(str(node['answer'] + '\n'))
        file.write('\n')
    for node in nodes:
        question = propose_prompt.format(input = node['answer'], k = k)
        response = llm.chat.completions.create(
            model = model,
            temperature = temperature,
            messages = [
                {
                    'role': 'user',
                    'content': question,
                }
            ],
        )
        print('generator response: ' + response.choices[0].message.content)
        answers = Parse_propose_response(response.choices[0].message.content)
        for answer in answers:
            new_nodes.append({'id': id, 'answer': answer, 'value': None, 'parent_node': node['id'], 'ancestor_value': Value_mapping(node['value']) + node['ancestor_value']})
            id += 1
    return new_nodes

def Evaluator(llm: OpenAI, model, temperature, nodes):
    new_nodes = list()
    print('Evaluator: ')
    for node in nodes:
        string = re.search('left.+', node['answer']).group().replace('left: ', '').replace(')', '')
        print('evaluator question: ' + string)
        question = value_prompt.format(input = string, k = k)
        response = llm.chat.completions.create(
            model = model,
            temperature = temperature,
            messages = [
                {
                    'role': 'user',
                    'content': question,
                }
            ],
        )
        print('evaluator response: ' + response.choices[0].message.content)
        value = Parse_value_response(response.choices[0].message.content)
        node['value'] = value
        new_nodes.append(node)
    return new_nodes
        

def Value_mapping(value):
    order = {'sure': 20, 'likely': 1, 'impossible': 0.001, None: 0}
    return order.get(value, float(0))

def Sorted_by_value(node):
    return Value_mapping(node['value']) + node['ancestor_value']

def Sorted_by_id(node):
    return node['id']

def Get_answer_by_cot(llm: OpenAI, temperature, model, input, answer):
    string = input + '\nSteps:\n' + answer
    print(string)
    question = cot_prompt.format(input = string, k = k)
    response = llm.chat.completions.create(
        model = model,
        temperature = temperature,
        messages = [
            {
                'role': 'user',
                'content': question,
            }
        ],
    )
    print('cot final: ' + response.choices[0].message.content)
    final_answer = response.choices[0].message.content
    return final_answer.lower().replace('answer: ', '').split('=')[0]

def Acc(answer):
    return bool(sympy.simplify(answer) == 24)
    
if __name__ == '__main__':
    # initializing
    dotenv.load_dotenv()
    llm = OpenAI(
        api_key = os.getenv('OPENAI_API_KEY'),
    )
    #model = 'gpt-3.5-turbo-1106'
    model = 'gpt-4-0613'
    temperature = 0.7
    with open('record.txt', 'w'):
        pass
    with open('24.json', 'w'):
        pass
    data = Load_data()
    puzzles_id = 0
    k = 5
    T = 3
    b = 3
    
    states = list()
    for j in range(1):
        Nodes = [{'id': 0, 'answer': data[puzzles_id], 'value': None, 'parent_node': None, 'ancestor_value': 0}]
        Top_b = [Nodes[0]]
        steps = list()
        print(f'Task {puzzles_id}')
        with open('record.txt', 'a') as file:
            file.write(f'Task {puzzles_id}\n')
        for i in range(T):
            new_nodes = Generator(llm, model, temperature, Top_b, k, len(Nodes))
            for node in new_nodes:
                print(node['answer'], node['parent_node'])
            new_nodes = Evaluator(llm, model, temperature, new_nodes)
            new_nodes = sorted(new_nodes, key = Sorted_by_value, reverse = True)
            with open('record.txt', 'a') as file:
                file.write('\nnew nodes:\n')
            for node in new_nodes:
                print(node['answer'] + ' ' + node['value'] + ' ' + str(node['parent_node']))
                with open('record.txt', 'a') as file:
                    file.write(str(node['id']) + ' ' + node['answer'] + ' ' + str(Value_mapping(node['value']) + node['ancestor_value']) + ' ' + str(node['parent_node']) + '\n')
            Top_b = new_nodes[:b]
            print(f'Top {b} nodes')
            with open('record.txt', 'a') as file:
                file.write(f'Top {b} nodes:\n')
            for item in Top_b:
                print(item['id'])
                with open('record.txt', 'a') as file:
                    file.write(str(item['id']) + '\n')
            print('new nodes')
            for node in new_nodes:
                print(node)
                Nodes.append(node)
            Nodes = sorted(Nodes, key = Sorted_by_id)
            steps.append({'step': i, 'Nodes': Nodes.copy(), 'Top_b': Top_b})
            print(steps)
            
        best = Top_b[0]
        answer = ''
        while best['parent_node'] != None:
            answer = best['answer'] + '\n' + answer
            best = Nodes[best['parent_node']]
        final_answer = Get_answer_by_cot(llm, temperature, model, data[puzzles_id], answer)
        question = {'id': puzzles_id, 'steps': steps, 'answer': final_answer, 'correct': Acc(final_answer)}
        states.append(question)
        with open('record.txt', 'a') as file:
            file.write('final answer: ' + final_answer + '\n')
        puzzles_id += 1
    
    with open('24.json', 'a') as file:
        json.dump(states, file, indent = 4)
    
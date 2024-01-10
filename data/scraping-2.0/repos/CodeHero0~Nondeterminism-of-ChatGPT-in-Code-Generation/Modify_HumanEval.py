import json
import os
import openai
import re
import subprocess
log_file = './log/demo.log'
problem_list = []
model = 'gpt-3.5-turbo'
topn = 5
temperature = float(1)
openai.api_key = ''
# for_list = [32, 38, 44, 50, 53, 65, 66, 68, 71, 74, 76, 79, 83, 84, 88, 89, 91, 92, 93, 94, 100, 101, 105, 107, 108, 109, 111, 113, 114, 115, 116, 122, 123, 126, 128, 132, 133, 140, 143, 145, 151, 152, 154, 157, 159, 160, 163]
def run_test_case(i):
    test_cases = test_case_dic[problem_list[i]['task_id']]
    demo_file = 'demo.py'
    with open(demo_file, 'w') as f:
        f.write(problem_list[i]['prompt'] + problem_list[i]['canonical_solution'])
    call_demo_file = 'call_demo.py'
    unpassed_test_case = []
    for j in range(len(test_cases)):
        if test_cases[j]['relation'] == '==':
            with open('./call_demo.py', 'w') as f:
                f.write('from %s import %s\nprint(%s(%s))' % (
                    demo_file.split('.')[0],
                    problem_list[i]['entry_point'],
                    problem_list[i]['entry_point'],
                    test_cases[j]['input']
                ))
            try:
                output = subprocess.run(["python", call_demo_file], capture_output=True, text=True, timeout=3)

            except subprocess.TimeoutExpired as e:
                print(e, flush=True)
                unpassed_test_case.append([j, 'Timeout'])
                continue
            except Exception as e:
                print(e, flush=True)
                unpassed_test_case.append([j, 'Exception'])
                continue
            if test_cases[j]['output'].strip() != output.stdout.strip():
                unpassed_test_case.append([j, 'false'])
            else:
                unpassed_test_case.append([j, 'True'])
        else:
            if '$input$' in test_cases[j]['relation'] or '$demo$' in test_cases[j]['relation']:
                with open('./call_demo.py', 'w') as f:
                    f.write('from %s import %s\n%s' % (
                        demo_file.split('.')[0],
                        problem_list[i]['entry_point'],
                        test_cases[j]['relation'].replace('$input$', str(test_cases[j]['input'])).replace('$demo$', demo_file.split('.')[0])
                    ))
            else:
                with open('./call_demo.py', 'w') as f:
                    f.write('from %s import %s\nprint(%s)' % (demo_file.split('.')[0],
                        problem_list[i]['entry_point'],
                        test_cases[j]['relation'].replace('candidate', problem_list[i]['entry_point'])))
            try:
                output = subprocess.run(["python", call_demo_file], capture_output=True, text=True, timeout=3)
            except subprocess.TimeoutExpired as e:
                print(e, flush=True)
                unpassed_test_case.append([j, 'Timeout'])
                continue
            except Exception as e:
                print(e, flush=True)
                unpassed_test_case.append([j, 'Exception'])
                continue
            if output.stdout.strip() != 'True':
                unpassed_test_case.append([j, 'false'])
            else:
                unpassed_test_case.append([j, 'True'])
    if len(set([i[1] for i in unpassed_test_case])) == 1 and unpassed_test_case[0][1] == 'True':
        print('ALL TRUE')
    print(unpassed_test_case)

def description_2_code(description, model, topn, temperature):
    prompt = 'Generate Python3 code (Markdown):\n'
    completion = openai.ChatCompletion.create(
        model=model,
        n=topn,
        temperature=temperature,
        messages=[{"role": "user",
                   "content": prompt + description},
                  ]
    )
    response_list = []
    for i in completion['choices']:
        response_list.append(i['message']['content'])
    return response_list

with open('./HumanEval/HumanEval.jsonl', 'r') as f:
    for line in f.readlines():
        problem_list.append(json.loads(line))

def demo():
    i = 0
    while not input():
        print(problem_list[i]['task_id'])
        print(problem_list[i]['test'])
        # print(problem_list[i]['prompt'] + problem_list[i]['canonical_solution']+problem_list[i]['test'])
        i += 1

def test(i):
    print(problem_list[i]['task_id'])
    print(problem_list[i]['test'])


test_case_dic = {}
p = r'candidate\((.*?)\) == (.*)'
# p = r'candidate\((.*?)\)\s+==\s+(.*)'
for problem in problem_list:
    name = problem['task_id']
    testcase = []
    for line in problem['test'].split('assert'):
        if 'candidate(' in line:
            # input
            p1 = re.search(p, line)
            if p1:
                # print('input:'+p1.group(1))
                # print('output:'+p1.group(2))
                output = p1.group(2).strip()
                if ('\'' in output[0] and '\'' in output[-1]) or \
                        ('\"' in output[0] and '\"' in output[-1]):
                    res = {
                        'input': p1.group(1),
                        'output': output[1:-1],
                        'relation': '=='
                    }
                else:
                    res = {
                        'input': p1.group(1),
                        'output': p1.group(2),
                        'relation': '=='
                    }
                testcase.append(res)
    test_case_dic[name] = testcase

test_case_dic['HumanEval/1'] = [
    {'input': "'(()()) ((())) () ((())()())'", 'output': "['(()())', '((()))', '()', '((())()())']", 'relation': '=='},
    {'input': "'() (()) ((())) (((())))'", 'output': "['()', '(())', '((()))', '(((())))']", 'relation': '=='},
    {'input': "'(()(())((())))'", 'output': "['(()(())((())))']", 'relation': '=='},
    {'input': "'( ) (( )) (( )( ))'", 'output': "['()', '(())', '(()())']", 'relation': '=='}
]
test_case_dic['HumanEval/2'] = [
    {'input': '3.5', 'output': '0.5', 'relation': '=='},
    {'input': '1.33', 'output': '1e-6', 'relation': 'abs(candidate(1.33) - 0.33) < 1e-6'},
    {'input': '123.456', 'output': '1e-6', 'relation': 'abs(candidate(123.456) - 0.456) < 1e-6'}
]
test_case_dic['HumanEval/4'] = [
    {'input': '[1.0, 2.0, 3.0]', 'output': '1e-6', 'relation': 'abs(candidate([1.0, 2.0, 3.0]) - 2.0/3.0) < 1e-6'},
    {'input': '[1.0, 2.0, 3.0, 4.0]', 'output': '1e-6', 'relation': 'abs(candidate([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-6'},
    {'input': '[1.0, 2.0, 3.0, 4.0, 5.0]', 'output': '1e-6', 'relation': 'abs(candidate([1.0, 2.0, 3.0, 4.0, 5.0]) - 6.0/5.0) < 1e-6'}
]
test_case_dic['HumanEval/8'][3] = {'input': '[3, 5, 7]', 'output': '(15, 105)', 'relation': '=='}
test_case_dic['HumanEval/33'] = [
    {'input': '[5, 6, 3, 4, 8, 9, 2]', 'output': '[2, 6, 3, 4, 8, 9, 5]', 'relation': '=='},
    {'input': '[5, 8, 3, 4, 6, 9, 2]', 'output': '[2, 8, 3, 4, 6, 9, 5]', 'relation': '=='},
    {'input': '[5, 6, 9, 4, 8, 3, 2]', 'output': '[2, 6, 9, 4, 8, 3, 5]', 'relation': '=='},
    {'input': '[5, 6, 3, 4, 8, 9, 2, 1]', 'output': '[2, 6, 3, 4, 8, 9, 5, 1]', 'relation': '=='}
]
test_case_dic['HumanEval/37'] = [
    {'input': '[1, 2, 3]', 'output': '[1, 2, 3]', 'relation': '=='},
    {'input': '[5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10]', 'output': '[-10, 3, -5, 2, -3, 3, 5, 0, 9, 1, 123]', 'relation': '=='},
    {'input': '[5, 8, -12, 4, 23, 2, 3, 11, 12, -10]', 'output': '[-12, 8, 3, 4, 5, 2, 12, 11, 23, -10]', 'relation': '=='}
]
test_case_dic['HumanEval/32'] = [{'input': [-10, -2], 'output': 1.1641532182693481e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-3, -6, -7, 7], 'output': 9.76619674020185e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [8, 3], 'output': 5.820766091346741e-11, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-10, -8], 'output': 4.656612873077393e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-3, 6, 9, -10], 'output': 1.337379096355562e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [10, 7, 3, -3], 'output': 1.3840022461408807e-09, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [8, -2, -10, -5, 3, 1, -2, -6], 'output': 6.92455426332117e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [1, -7, -8, 2], 'output': 2.1342083655895294e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [1, 1], 'output': 0.0, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-9, 4, 7, -7, 2, -8], 'output': 1.1405965061328516e-09, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [10, 9, 1, 8, -4, -8], 'output': 4.0877967677488414e-11, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-3, -1], 'output': 5.820766091346741e-11, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-3, -7], 'output': 5.820766091346741e-11, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-2, 4, 10, 1, -5, 1, 1, -4], 'output': 4.5996983999430086e-11, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [10, -8, 9, 10, -5, 7], 'output': 4.412106235918145e-09, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-5, 4, 2, -2], 'output': 7.292131343206165e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [1, -9, -3, -9], 'output': 1.7145054993783493e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [2, -2, -8, -4, 8, 1], 'output': 3.6866111552402714e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [10, 5, 2, 10], 'output': 1.015466821741029e-09, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-6, -2, -6, -3, 7, 7, -2, 8], 'output': 2.469873194854699e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [8, 2, 1, -3, -6, 6, 5, -8], 'output': 4.654125973502232e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-7, -6], 'output': 1.1641532182693481e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [3, 9, -8, 2], 'output': 4.748736473492166e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [9, 4, 6, -2, 7, -10, -7, 7], 'output': 1.0656506788109255e-09, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [10, 1, -7, -1, 3, -5], 'output': 6.19443163429878e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-10, -2, 6, -5, 6, -7, 10, -1], 'output': 1.039987151951749e-09, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-6, 1, -5, 7], 'output': 8.558842523598287e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [9, 1], 'output': 5.820766091346741e-11, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-10, -7, 1, -1, -3, -9, -3, 8], 'output': 9.059419880941277e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-8, 5], 'output': 1.1641532182693481e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [7, -6], 'output': 2.3283064365386963e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [5, 7, -5, -2], 'output': 3.864730757641155e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-4, 7, -4, -1, 2, 10, 1, 4], 'output': 1.152398176884617e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-7, -3, -3, -8, 1, -10, 8, 7], 'output': 1.1465629556894896e-09, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [8, -3, -10, -8], 'output': 8.052962741089686e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-3, -8], 'output': 4.656612873077393e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [1, -8], 'output': 4.656612873077393e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-2, 5, -4, 7], 'output': 2.8748137204104296e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [8, 8, 5, -3], 'output': 7.751452812954085e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [3, -4, -7, -7, 3, 1, 3, 3], 'output': 3.0882091502093534e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-9, 10, 10, -7, -9, 2, 1, -7], 'output': 2.323840675444444e-09, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-4, -4, 7, 4], 'output': 0.0, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [3, -5, -2, 4], 'output': 2.471778337564956e-11, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-8, 4, 7, -7], 'output': 5.787530454881562e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [10, 7], 'output': 5.820766091346741e-11, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-8, -3], 'output': 5.820766091346741e-11, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [3, 5, 5, -4], 'output': 4.028066769024008e-11, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-9, -5, 2, -10, 2, -2, 4, -1], 'output': 1.2186199688235533e-09, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [7, 5, -6, -4, -1, -4, -9, 8], 'output': 7.55201901014857e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [1, -9], 'output': 4.0745362639427185e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [8, 5], 'output': 1.7462298274040222e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-9, 6, -8, -5], 'output': 7.17989223630866e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [9, -8], 'output': 4.656612873077393e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [2, -7, 8, -3], 'output': 1.2934986415302774e-11, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [9, -8], 'output': 4.656612873077393e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [8, 8, 6, 1, -2, -4, 1, -3], 'output': 8.968825682131865e-11, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [2, -6, 10, -1, 4, 1], 'output': 1.2246800906723365e-08, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-10, 4], 'output': 2.3283064365386963e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-8, 7], 'output': 1.1641532182693481e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [6, -2, -6, 1], 'output': 4.1145209461745935e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-3, 1], 'output': 5.820766091346741e-11, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-5, 4, 7, -1, 9, 10], 'output': 2.8451518918615193e-11, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [7, -1], 'output': 5.820766091346741e-11, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-6, -2], 'output': 1.1641532182693481e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-7, 7], 'output': 4.0745362639427185e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-2, -1, 9, -4], 'output': 5.314582107729393e-12, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-4, 10, -2, 6, 5, -2], 'output': 5.341000801351026e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-8, 10], 'output': 1.1641532182693481e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-2, -9, -10, 1, -6, 10, -2, -5], 'output': 1.4370016288012266e-08, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [7, 3, 7, -10, -7, -8, -6, 7], 'output': 1.0816925133383393e-09, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [1, 8], 'output': 4.656612873077393e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [3, -6, -9, -1], 'output': 4.090063773776187e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-9, 1, -4, -3, -7, 1], 'output': 6.964910426177084e-08, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [9, -6, -3, -5, -5, 3, -10, -5], 'output': 1.3005894139439533e-09, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [3, -3, -2, -5, -7, 2], 'output': 0.0, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [5, -3], 'output': 1.1641532182693481e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [4, 1, -1, -3], 'output': 1.2522427539352066e-11, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-10, -4, 2, 1], 'output': 7.0775918459276e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-8, -2, 1, 10, 6, 2], 'output': 1.0347153134304676e-09, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-10, -7, -2, -5, 8, -2], 'output': 4.458877711499554e-12, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-7, 9], 'output': 2.3283064365386963e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [1, 1, 3, 9, 6, -7, 2, 8], 'output': 6.708447131131834e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-2, -9, 3, -10], 'output': 1.3271347909515896e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [1, 3, -8, 1], 'output': 9.151792171313566e-11, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-7, -1, 6, -1, 3, 1], 'output': 9.165997960636219e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-1, 7, -6, -4, 3, 2, -5, 9], 'output': 1.2270528522298832e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [2, 7, -10, -1, -1, -4], 'output': 8.104050763790838e-11, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [8, 9, 10, 1, 4, 4, 4, -4], 'output': 2.9445686777762603e-08, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-5, -8, -1, 6, 10, 9, 1, -8], 'output': 2.796114451086851e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-1, -3, -4, -6], 'output': 8.562428543967826e-11, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-9, -3], 'output': 1.7462298274040222e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [9, -8, 4, 3, 10, 8, -4, 2], 'output': 4.614358672938579e-09, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [2, -3, -6, 10, -10, -7, 3, -3], 'output': 2.5733340805467186e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [6, 4, -9, 7], 'output': 4.689382215872229e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-7, 4, -6, 4], 'output': 9.2210683533267e-12, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [4, 9, 6, 3, 7, 4], 'output': 2.5149304860860866e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [5, 4, -2, -3], 'output': 1.9339907453286287e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [6, 5, 10, -3, -2, 4], 'output': 1.9849579757647007e-09, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [-1, -3], 'output': 1.1641532182693481e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}, {'input': [1, 1, 7, -8, -6, -6], 'output': 4.970059919173764e-10, 'relation': 'from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)'}]
test_case_dic['HumanEval/38'] = [{'input': "'axdhhixdexrvsncacbgh'", 'output': 'daxihhexdvxrcsnbacgh', 'relation': '=='}, {'input': "'artwugrnwoshzaizfy'", 'output': 'targwuwrnhosizayzf', 'relation': '=='}, {'input': "'iekykgcmdlldiztb'", 'output': 'kiegykdcmdlltizb', 'relation': '=='}, {'input': "'dmrrjctlugwsbvchy'", 'output': 'rdmcrjutlsgwcbvhy', 'relation': '=='}, {'input': "'hdciomlfulglvi'", 'output': 'chdmioulfllgvi', 'relation': '=='}, {'input': "'ctufruhfxmiowruvkhyy'", 'output': 'uctufrxhfomiuwrhvkyy', 'relation': '=='}, {'input': "'bzhmikgscw'", 'output': 'hbzkmicgsw', 'relation': '=='}, {'input': "'upguomieexrhixr'", 'output': 'gupmuoeiehxrrix', 'relation': '=='}, {'input': "'smnhelpcqbdyufevnzt'", 'output': 'nsmlheqpcybdeufzvnt', 'relation': '=='}, {'input': "'mtmqioavrxd'", 'output': 'mmtoqiravxd', 'relation': '=='}, {'input': "'yirukyjndoafxixyfqqd'", 'output': 'ryiyukdjnfoaxxiqyfqd', 'relation': '=='}, {'input': "'uqjgetyflyqrtkaadplz'", 'output': 'juqtgelyfryqatkpadlz', 'relation': '=='}, {'input': "'bhhccspcxryyee'", 'output': 'hbhsccxpcyryee', 'relation': '=='}, {'input': "'rfpqtigrnxwywjgvumlo'", 'output': 'prfiqtngryxwgwjmvulo', 'relation': '=='}, {'input': "'dhockhsrashhcwabhu'", 'output': 'odhhckasrhshacwubh', 'relation': '=='}, {'input': "'kcbhiqpgvre'", 'output': 'bkcqhivpgre', 'relation': '=='}, {'input': "'phspzzgdnvndnnlxbov'", 'output': 'sphzpzngddvnlnnoxbv', 'relation': '=='}, {'input': "'dbuxkmdhzgrgenoiofhc'", 'output': 'udbmxkzdhggroenfiohc', 'relation': '=='}, {'input': "'rdzurbcyafnhpgpmb'", 'output': 'zrdburacyhfnppgmb', 'relation': '=='}, {'input': "'ammzzijnoxzw'", 'output': 'mamizzojnwxz', 'relation': '=='}, {'input': "'wpvgjebsgrbxkbxspb'", 'output': 'vwpegjgbsxrbxkbbsp', 'relation': '=='}, {'input': "'fbqcfqtcchmvshdtbs'", 'output': 'qfbqcfctcvhmdshstb', 'relation': '=='}, {'input': "'nvcsqsigkwkvimhvuej'", 'output': 'cnvssqkigvwkhimevuj', 'relation': '=='}, {'input': "'yckotadcsgqrelich'", 'output': 'kycaotsdcrgqielch', 'relation': '=='}, {'input': "'fojwjrzutavqjvr'", 'output': 'jforwjtzuqavrjv', 'relation': '=='}, {'input': "'idexrdijetg'", 'output': 'eiddxreijtg', 'relation': '=='}, {'input': "'vugqpibciniuakb'", 'output': 'gvuiqpibcunibak', 'relation': '=='}, {'input': "'ifuorxnrwdca'", 'output': 'uifxorwnradc', 'relation': '=='}, {'input': "'blrresebnlzj'", 'output': 'rblsrenebjlz', 'relation': '=='}, {'input': "'gvlvdhyrln'", 'output': 'lgvhvdlyrn', 'relation': '=='}, {'input': "'ehxzzfnafxkfnzzxzvh'", 'output': 'xehfzzfnafxkznzvxzh', 'relation': '=='}, {'input': "'zwfmbdhgpljozh'", 'output': 'fzwdmbphgoljzh', 'relation': '=='}, {'input': "'vgakimyicuqlm'", 'output': 'avgmkicyiluqm', 'relation': '=='}, {'input': "'karifdibstndxzlntkqd'", 'output': 'rkadifsibdtnlxzkntqd', 'relation': '=='}, {'input': "'giswnbqzavxrxvxg'", 'output': 'sgibwnaqzrvxxxvg', 'relation': '=='}, {'input': "'cvntkkdxvqjjnkv'", 'output': 'ncvktkvdxjqjvnk', 'relation': '=='}, {'input': "'jrwgnemvvftxjmsr'", 'output': 'wjregnvmvxftsjmr', 'relation': '=='}, {'input': "'jgjzsnukto'", 'output': 'jjgnzstuko', 'relation': '=='}, {'input': "'vgopzqxfzcjvvuqtk'", 'output': 'ovgqpzzxfvcjqvutk', 'relation': '=='}, {'input': "'hvyhzjeagbh'", 'output': 'yhvjhzgeabh', 'relation': '=='}, {'input': "'yctnuogwsmpwhemuw'", 'output': 'tyconusgwwmpmheuw', 'relation': '=='}, {'input': "'ydynhyzwfq'", 'output': 'yydynhfzwq', 'relation': '=='}, {'input': "'rhboedovzrtqyoktx'", 'output': 'brhdoezovqrtkyotx', 'relation': '=='}, {'input': "'ronxpfiyouihyqyuhp'", 'output': 'nrofxpoiyhuiyyqpuh', 'relation': '=='}, {'input': "'cwohijkrkeechm'", 'output': 'ocwjhikkrceehm', 'relation': '=='}, {'input': "'gcwnknonrgnb'", 'output': 'wgcnnkronbgn', 'relation': '=='}, {'input': "'swyysapamjylnrmx'", 'output': 'yswaysmpaljymnrx', 'relation': '=='}, {'input': "'thzhippankvmzmvfox'", 'output': 'zthphinpamkvvzmxfo', 'relation': '=='}, {'input': "'ratssmacvneu'", 'output': 'tramssvacune', 'relation': '=='}, {'input': "'bifkgmkkomiyniycp'", 'output': 'fbimkgokkymiynicp', 'relation': '=='}, {'input': "'rbxhulyucb'", 'output': 'xrblhucyub', 'relation': '=='}, {'input': "'gahehtpved'", 'output': 'hgatehepvd', 'relation': '=='}, {'input': "'owgylittfwdxfjysadj'", 'output': 'gowiylfttxwdyfjdsaj', 'relation': '=='}, {'input': "'mmvgcwwusdwhjvyzdtz'", 'output': 'vmmwgcswuhdwyjvtzdz', 'relation': '=='}, {'input': "'blznvrcqlkaupdnluno'", 'output': 'zblrnvlcqukanpdnluo', 'relation': '=='}, {'input': "'fxnuiqzrtpoy'", 'output': 'nfxquitzrypo', 'relation': '=='}, {'input': "'sixhckohiosyvmtk'", 'output': 'xsikhciohyostvmk', 'relation': '=='}, {'input': "'kfpglpikzi'", 'output': 'pkfpglziki', 'relation': '=='}, {'input': "'irwqgahxcprnhwyuwpp'", 'output': 'wiraqgchxnpryhwpuwp', 'relation': '=='}, {'input': "'aczhmjhjwslvrqpln'", 'output': 'zacjhmwhjvslprqln', 'relation': '=='}, {'input': "'lwkijohdigkxxrdwfy'", 'output': 'klwoijihdxgkdxrywf', 'relation': '=='}, {'input': "'xpgxsiqtydgjj'", 'output': 'gxpixsyqtjdgj', 'relation': '=='}, {'input': "'fjlwraiberjbw'", 'output': 'lfjawreibbrjw', 'relation': '=='}, {'input': "'ypuasdppjkfo'", 'output': 'uypdasjppokf', 'relation': '=='}, {'input': "'pdimpcsucv'", 'output': 'ipdcmpcsuv', 'relation': '=='}, {'input': "'ezejcsdrhy'", 'output': 'eezsjchdry', 'relation': '=='}, {'input': "'tzthytmoqjsojsnt'", 'output': 'ttzthyqmoojsnjst', 'relation': '=='}, {'input': "'xdtguyivgc'", 'output': 'txdygugivc', 'relation': '=='}, {'input': "'frhfacownpjt'", 'output': 'hfrcfanowtpj', 'relation': '=='}, {'input': "'jwhwojvhci'", 'output': 'hjwjwocvhi', 'relation': '=='}, {'input': "'vzsndghurieebfcjtzxs'", 'output': 'svzgndrhueiecbfzjtxs', 'relation': '=='}, {'input': "'doojwwiqmporct'", 'output': 'odowjwmiqrpoct', 'relation': '=='}, {'input': "'xkniathvcs'", 'output': 'nxktiachvs', 'relation': '=='}, {'input': "'yvasbiyfyqupifonusp'", 'output': 'ayvisbyyfpquoifsnup', 'relation': '=='}, {'input': "'lnpkvkfkdnw'", 'output': 'plnkkvdfknw', 'relation': '=='}, {'input': "'vmjrbyckokdimqyav'", 'output': 'jvmyrbockikdymqav', 'relation': '=='}, {'input': "'nboqlgyptoyugibejr'", 'output': 'onbgqltypuoybgirej', 'relation': '=='}, {'input': "'pdwutahwzjrfrnach'", 'output': 'wpdautzhwfjrarnch', 'relation': '=='}, {'input': "'duopweqwjin'", 'output': 'oduepwjqwin', 'relation': '=='}, {'input': "'hopemrtqgecxyzink'", 'output': 'phoremgtqxeciyznk', 'relation': '=='}, {'input': "'ajijsxvpsorelkpyrr'", 'output': 'iajxjssvpeorplkryr', 'relation': '=='}, {'input': "'kgohswhymbknpwxz'", 'output': 'okgwhsmhynbkxpwz', 'relation': '=='}, {'input': "'vzmepueqbkdsdqoo'", 'output': 'mvzuepbeqskdodqo', 'relation': '=='}, {'input': "'enxecuzipk'", 'output': 'xenuecpzik', 'relation': '=='}, {'input': "'muwkvcmkrwyurbpchtu'", 'output': 'wmuckvrmkuwyprbtchu', 'relation': '=='}, {'input': "'hxjndcuwyofdjawkzbbj'", 'output': 'jhxcndyuwdofwjabkzbj', 'relation': '=='}, {'input': "'nelqnhvzsffftmc'", 'output': 'lnehqnsvzfffctm', 'relation': '=='}, {'input': "'hpvehsuioivozoavrjf'", 'output': 'vhpsehouioivazojvrf', 'relation': '=='}, {'input': "'lsounjiowjg'", 'output': 'olsjunwiojg', 'relation': '=='}, {'input': "'dhpslmjwsavjiams'", 'output': 'pdhmslsjwjavmias', 'relation': '=='}, {'input': "'xbyxptyzjtzhhultigvy'", 'output': 'yxbtxpjyzhtzlhugtivy', 'relation': '=='}, {'input': "'euvuudjzbbsoxeljkcxn'", 'output': 'veuduubjzobslxecjkxn', 'relation': '=='}, {'input': "'ezglqrifqpzi'", 'output': 'gezrlqqifipz', 'relation': '=='}, {'input': "'kzxocdyhexvvmz'", 'output': 'xkzdoceyhvxvmz', 'relation': '=='}, {'input': "'czlaimdorvxlisvulm'", 'output': 'lczmairdolvxvismul', 'relation': '=='}, {'input': "'hpvtrathkuc'", 'output': 'vhpatrkthuc', 'relation': '=='}, {'input': "'wjondubbepdjhrdmoelv'", 'output': 'owjundebbjpddhremolv', 'relation': '=='}, {'input': "'sxnenxdpunitwlboog'", 'output': 'nsxxenudptnibwlgoo', 'relation': '=='}, {'input': "'dvlrulbmlgdio'", 'output': 'ldvlrulbmigdo', 'relation': '=='}, {'input': "'guvtauzkbhe'", 'output': 'vguutabzkhe', 'relation': '=='}]
test_case_dic['HumanEval/44'] = test_case_dic['HumanEval/44'][:-1]
for x in range(2, 8):
    test_case_dic['HumanEval/44'].append({
        'input': '%s, %s' % (x, x+1),
        'output': '%s' % (x),
        'relation': '=='
    })
test_case_dic['HumanEval/52'] = [
    {'input': '[1, 2, 4, 10], 100', 'output': 'True', 'relation': '=='},
    {'input': '[1, 20, 4, 10], 5', 'output': 'False', 'relation': '=='},
    {'input': '[1, 20, 4, 10], 21', 'output': 'True', 'relation': '=='},
    {'input': '[1, 20, 4, 10], 22', 'output': 'True', 'relation': '=='},
    {'input': '[1, 8, 4, 10], 11', 'output': 'True', 'relation': '=='},
    {'input': '[1, 8, 4, 10], 10', 'output': 'False', 'relation': '=='}
]
test_case_dic['HumanEval/50'] = [{'input': "'ifcnmmjciacwhxsgfhlm'", 'output': 'daxihhexdvxrcsnbacgh', 'relation': '=='}, {'input': "'yfwlbzbwsmtxnefdek'", 'output': 'targwuwrnhosizayzf', 'relation': '=='}, {'input': "'pnjldpihriqqyneg'", 'output': 'kiegykdcmdlltizb', 'relation': '=='}, {'input': "'wirhwozyqxlbhgamd'", 'output': 'rdmcrjutlsgwcbvhy', 'relation': '=='}, {'input': "'hmirntzqkqqlan'", 'output': 'chdmioulfllgvi', 'relation': '=='}, {'input': "'zhyzkwcmktrnzbwmapdd'", 'output': 'uctufrxhfomiuwrhvkyy', 'relation': '=='}, {'input': "'mgeprnhlxb'", 'output': 'hbzkmicgsw', 'relation': '=='}, {'input': "'lzurztjnjmcwwnc'", 'output': 'gupmuoeiehxrrix', 'relation': '=='}, {'input': "'sxrqmjvuhdgijzkeasy'", 'output': 'nsmlheqpcybdeufzvnt', 'relation': '=='}, {'input': "'rrytvnwfaci'", 'output': 'mmtoqiravxd', 'relation': '=='}, {'input': "'wdndzpiosktfccnvdkvi'", 'output': 'ryiyukdjnfoaxxiqyfqd', 'relation': '=='}, {'input': "'ozvyljqdkwdvfypufiqe'", 'output': 'juqtgelyfryqatkpadlz', 'relation': '=='}, {'input': "'mgmxhhcuhdwdjj'", 'output': 'hbhsccxpcyryee', 'relation': '=='}, {'input': "'uwknvyslwdcblborazqt'", 'output': 'prfiqtngryxwgwjmvulo', 'relation': '=='}, {'input': "'timmhpfxwmxmfhbzgm'", 'output': 'odhhckasrhshacwubh', 'relation': '=='}, {'input': "'gphvmnaulwj'", 'output': 'bkcqhivpgre', 'relation': '=='}, {'input': "'xumeuesliiasqsstcga'", 'output': 'sphzpzngddvnlnnoxbv', 'relation': '=='}, {'input': "'zigrcpeimllwtjskntmh'", 'output': 'udbmxkzdhggroenfiohc', 'relation': '=='}, {'input': "'ewigzwfhdmksuulrg'", 'output': 'zrdburacyhfnppgmb', 'relation': '=='}, {'input': "'rfrneetosbce'", 'output': 'mamizzojnwxz', 'relation': '=='}, {'input': "'abujlolgxcwgcpggxu'", 'output': 'vwpegjgbsxrbxkbbsp', 'relation': '=='}, {'input': "'vkgvhkhyhamrixmxyg'", 'output': 'qfbqcfctcvhmdshstb', 'relation': '=='}, {'input': "'hsaxxvpnlabpmnrjazo'", 'output': 'cnvssqkigvwkhimevuj', 'relation': '=='}, {'input': "'pdhftyxihwlvnjqhm'", 'output': 'kycaotsdcrgqielch', 'relation': '=='}, {'input': "'oktwboyezvfawoa'", 'output': 'jforwjtzuqavrjv', 'relation': '=='}, {'input': "'jniicwjnoyl'", 'output': 'eiddxreijtg', 'relation': '=='}, {'input': "'laznvunghzsngfp'", 'output': 'gvuiqpibcunibak', 'relation': '=='}, {'input': "'znkctwbswfih'", 'output': 'uifxorwnradc', 'relation': '=='}, {'input': "'wgqxwjsjgoqe'", 'output': 'rblsrenebjlz', 'relation': '=='}, {'input': "'qlamaiqdws'", 'output': 'lgvhvdlyrn', 'relation': '=='}, {'input': "'cjmkeeksfkcpeseacem'", 'output': 'xehfzzfnafxkznzvxzh', 'relation': '=='}, {'input': "'kebirgumltqoem'", 'output': 'fzwdmbphgoljzh', 'relation': '=='}, {'input': "'falrpnhdnqzvr'", 'output': 'avgmkicyiluqm', 'relation': '=='}, {'input': "'wpfinkxngiysqcepsyvi'", 'output': 'rkadifsibdtnlxzkntqd', 'relation': '=='}, {'input': "'xlngbsfvewacccal'", 'output': 'sgibwnaqzrvxxxvg', 'relation': '=='}, {'input': "'shapypaicovoasp'", 'output': 'ncvktkvdxjqjvnk', 'relation': '=='}, {'input': "'bowjlsarackyxorw'", 'output': 'wjregnvmvxftsjmr', 'relation': '=='}, {'input': "'oolsexyzpt'", 'output': 'jjgnzstuko', 'relation': '=='}, {'input': "'talvueeckahovazyp'", 'output': 'ovgqpzzxfvcjqvutk', 'relation': '=='}, {'input': "'dmaomeljfgm'", 'output': 'yhvjhzgeabh', 'relation': '=='}, {'input': "'ydhtszxlbbrurmjzb'", 'output': 'tyconusgwwmpmheuw', 'relation': '=='}, {'input': "'ddidsmkebv'", 'output': 'yydynhfzwq', 'relation': '=='}, {'input': "'gwmitjetavwypdtyc'", 'output': 'brhdoezovqrtkyotx', 'relation': '=='}, {'input': "'swtkcutndmznddvuzm'", 'output': 'nrofxpoiyhuiyyqpuh', 'relation': '=='}, {'input': "'thbomnppwhjjmr'", 'output': 'ocwjhikkrceehm', 'relation': '=='}, {'input': "'blhsspwtsgls'", 'output': 'wgcnnkronbgn', 'relation': '=='}, {'input': "'dxbfdxrufqodrswc'", 'output': 'yswaysmpaljymnrx', 'relation': '=='}, {'input': "'eymumnsufrpaaerckt'", 'output': 'zthphinpamkvvzmxfo', 'relation': '=='}, {'input': "'ywfrxxafhzsj'", 'output': 'tramssvacune', 'relation': '=='}, {'input': "'kgnrpltppdrndsnhu'", 'output': 'fbimkgokkymiynicp', 'relation': '=='}, {'input': "'cwgqmzhdzg'", 'output': 'xrblhucyub', 'relation': '=='}, {'input': "'mlfyjmjuai'", 'output': 'hgatehepvd', 'relation': '=='}, {'input': "'ltbndqkyycbidkoixfo'", 'output': 'gowiylfttxwdyfjdsaj', 'relation': '=='}, {'input': "'arrblhxbzmibdoayeie'", 'output': 'vmmwgcswuhdwyjvtzdz', 'relation': '=='}, {'input': "'egqwsaqhvzpfsuisqzt'", 'output': 'zblrnvlcqukanpdnluo', 'relation': '=='}, {'input': "'skcvznyewdut'", 'output': 'nfxquitzrypo', 'relation': '=='}, {'input': "'cxnpmhntmdtxyarp'", 'output': 'xsikhciohyostvmk', 'relation': '=='}, {'input': "'upkulqenpn'", 'output': 'pkfpglziki', 'relation': '=='}, {'input': "'bnwfvlhmcsuwdmbuzbu'", 'output': 'wiraqgchxnpryhwpuwp', 'relation': '=='}, {'input': "'efhomrbmoaxquwvqs'", 'output': 'zacjhmwhjvslprqln', 'relation': '=='}, {'input': "'pqbtnonmiclpicwdbk'", 'output': 'klwoijihdxgkdxrywf', 'relation': '=='}, {'input': "'lcuncxdvyoilo'", 'output': 'gxpixsyqtjdgj', 'relation': '=='}, {'input': "'qkofbwjnggwob'", 'output': 'lfjawreibbrjw', 'relation': '=='}, {'input': "'zduifxouutpk'", 'output': 'uypdasjppokf', 'relation': '=='}, {'input': "'nuihruhxza'", 'output': 'ipdcmpcsuv', 'relation': '=='}, {'input': "'jjexohmiwd'", 'output': 'eezsjchdry', 'relation': '=='}, {'input': "'yyeymdvrttoxsoxy'", 'output': 'ttzthyqmoojsnjst', 'relation': '=='}, {'input': "'ycidlzlnah'", 'output': 'txdygugivc', 'relation': '=='}, {'input': "'mkwhkfstbyuo'", 'output': 'hfrcfanowtpj', 'relation': '=='}, {'input': "'mobobthamn'", 'output': 'hjwjwocvhi', 'relation': '=='}, {'input': "'xaelsiwmzjnjhgkeoycx'", 'output': 'svzgndrhueiecbfzjtxs', 'relation': '=='}, {'input': "'titbobrnvwuthy'", 'output': 'odowjwmiqrpoct', 'relation': '=='}, {'input': "'scpynfhmax'", 'output': 'nxktiachvs', 'relation': '=='}, {'input': "'fdanxgddkuvztnkxszu'", 'output': 'ayvisbyyfpquoifsnup', 'relation': '=='}, {'input': "'uqsppaikpsb'", 'output': 'plnkkvdfknw', 'relation': '=='}, {'input': "'oardwgthpnpidrvfa'", 'output': 'jvmyrbockikdymqav', 'relation': '=='}, {'input': "'tsglvqyduztdglnwjo'", 'output': 'onbgqltypuoybgirej', 'relation': '=='}, {'input': "'buifzyembkowfwshm'", 'output': 'wpdautzhwfjrarnch', 'relation': '=='}, {'input': "'tizjubovbns'", 'output': 'oduepwjqwin', 'relation': '=='}, {'input': "'umtwjrlyvcjhndesp'", 'output': 'phoremgtqxeciyznk', 'relation': '=='}, {'input': "'nfocoxxaujtwuqpwdw'", 'output': 'iajxjssvpeorplkryr', 'relation': '=='}, {'input': "'tplbmxrmdsgpcube'", 'output': 'okgwhsmhynbkxpwz', 'relation': '=='}, {'input': "'raezjugjvxpitivt'", 'output': 'mvzuepbeqskdodqo', 'relation': '=='}, {'input': "'cjszjhuenp'", 'output': 'xenuecpzik', 'relation': '=='}, {'input': "'brzhpawrpzbduwgyhmz'", 'output': 'wmuckvrmkuwyprbtchu', 'relation': '=='}, {'input': "'omchsidzbitkbofgpego'", 'output': 'jhxcndyuwdofwjabkzbj', 'relation': '=='}, {'input': "'qsjmvsxaekkkhyr'", 'output': 'lnehqnsvzfffctm', 'relation': '=='}, {'input': "'amuxjmtzntnafetoawk'", 'output': 'vhpsehouioivazojvrf', 'relation': '=='}, {'input': "'tqxozsbntol'", 'output': 'olsjunwiojg', 'relation': '=='}, {'input': "'uimrxqxobofarnfx'", 'output': 'pdhmslsjwjavmias', 'relation': '=='}, {'input': "'dcgycuodemyeqmzlynad'", 'output': 'yxbtxpjyzhtzlhugtivy', 'relation': '=='}, {'input': "'ajzizzgoetgxqcjhopcs'", 'output': 'veuduubjzobslxecjkxn', 'relation': '=='}, {'input': "'ljewqvvnknue'", 'output': 'gezrlqqifipz', 'relation': '=='}, {'input': "'cpeithjdmacare'", 'output': 'xkzdoceyhvxvmz', 'relation': '=='}, {'input': "'qherfnwitqacanxrzq'", 'output': 'lczmairdolvxvismul', 'relation': '=='}, {'input': "'amufywpymzh'", 'output': 'vhpatrkthuc', 'relation': '=='}, {'input': "'tbozsijggouiimwjrtqa'", 'output': 'owjundebbjpddhremolv', 'relation': '=='}, {'input': "'sxccjsziuysngbqltt'", 'output': 'nsxxenudptnibwlgoo', 'relation': '=='}, {'input': "'qiaqwzqgrnlit'", 'output': 'ldvlrulbmigdo', 'relation': '=='}, {'input': "'alzzyfgepmj'", 'output': 'vguutabzkhe', 'relation': '=='}]
test_case_dic['HumanEval/51'][1] = {'input': '"abcdef\\nghijklm"', 'output': 'bcdf\nghjklm', 'relation': '=='}
test_case_dic['HumanEval/53'] = test_case_dic['HumanEval/53'][:-1] + [{'input': '654, 114', 'output': '768', 'relation': '=='}, {'input': '25, 759', 'output': '784', 'relation': '=='}, {'input': '281, 250', 'output': '531', 'relation': '=='}, {'input': '228, 142', 'output': '370', 'relation': '=='}, {'input': '754, 104', 'output': '858', 'relation': '=='}, {'input': '692, 758', 'output': '1450', 'relation': '=='}, {'input': '913, 558', 'output': '1471', 'relation': '=='}, {'input': '89, 604', 'output': '693', 'relation': '=='}, {'input': '432, 32', 'output': '464', 'relation': '=='}, {'input': '30, 95', 'output': '125', 'relation': '=='}, {'input': '223, 238', 'output': '461', 'relation': '=='}, {'input': '517, 616', 'output': '1133', 'relation': '=='}, {'input': '27, 574', 'output': '601', 'relation': '=='}, {'input': '203, 733', 'output': '936', 'relation': '=='}, {'input': '665, 718', 'output': '1383', 'relation': '=='}, {'input': '558, 429', 'output': '987', 'relation': '=='}, {'input': '225, 459', 'output': '684', 'relation': '=='}, {'input': '603, 284', 'output': '887', 'relation': '=='}, {'input': '828, 890', 'output': '1718', 'relation': '=='}, {'input': '6, 777', 'output': '783', 'relation': '=='}, {'input': '825, 163', 'output': '988', 'relation': '=='}, {'input': '714, 432', 'output': '1146', 'relation': '=='}, {'input': '348, 284', 'output': '632', 'relation': '=='}, {'input': '159, 220', 'output': '379', 'relation': '=='}, {'input': '980, 781', 'output': '1761', 'relation': '=='}, {'input': '344, 104', 'output': '448', 'relation': '=='}, {'input': '94, 389', 'output': '483', 'relation': '=='}, {'input': '99, 367', 'output': '466', 'relation': '=='}, {'input': '867, 352', 'output': '1219', 'relation': '=='}, {'input': '618, 270', 'output': '888', 'relation': '=='}, {'input': '826, 44', 'output': '870', 'relation': '=='}, {'input': '747, 470', 'output': '1217', 'relation': '=='}, {'input': '549, 127', 'output': '676', 'relation': '=='}, {'input': '996, 944', 'output': '1940', 'relation': '=='}, {'input': '387, 80', 'output': '467', 'relation': '=='}, {'input': '565, 300', 'output': '865', 'relation': '=='}, {'input': '849, 643', 'output': '1492', 'relation': '=='}, {'input': '633, 906', 'output': '1539', 'relation': '=='}, {'input': '882, 370', 'output': '1252', 'relation': '=='}, {'input': '591, 196', 'output': '787', 'relation': '=='}, {'input': '721, 71', 'output': '792', 'relation': '=='}, {'input': '46, 677', 'output': '723', 'relation': '=='}, {'input': '233, 791', 'output': '1024', 'relation': '=='}, {'input': '296, 81', 'output': '377', 'relation': '=='}, {'input': '875, 238', 'output': '1113', 'relation': '=='}, {'input': '887, 103', 'output': '990', 'relation': '=='}, {'input': '389, 284', 'output': '673', 'relation': '=='}, {'input': '464, 650', 'output': '1114', 'relation': '=='}, {'input': '854, 373', 'output': '1227', 'relation': '=='}, {'input': '166, 379', 'output': '545', 'relation': '=='}, {'input': '363, 214', 'output': '577', 'relation': '=='}, {'input': '686, 273', 'output': '959', 'relation': '=='}, {'input': '718, 959', 'output': '1677', 'relation': '=='}, {'input': '699, 663', 'output': '1362', 'relation': '=='}, {'input': '73, 623', 'output': '696', 'relation': '=='}, {'input': '650, 175', 'output': '825', 'relation': '=='}, {'input': '546, 746', 'output': '1292', 'relation': '=='}, {'input': '250, 167', 'output': '417', 'relation': '=='}, {'input': '473, 388', 'output': '861', 'relation': '=='}, {'input': '276, 947', 'output': '1223', 'relation': '=='}, {'input': '655, 704', 'output': '1359', 'relation': '=='}, {'input': '570, 224', 'output': '794', 'relation': '=='}, {'input': '701, 332', 'output': '1033', 'relation': '=='}, {'input': '863, 786', 'output': '1649', 'relation': '=='}, {'input': '794, 57', 'output': '851', 'relation': '=='}, {'input': '234, 841', 'output': '1075', 'relation': '=='}, {'input': '32, 824', 'output': '856', 'relation': '=='}, {'input': '323, 410', 'output': '733', 'relation': '=='}, {'input': '274, 67', 'output': '341', 'relation': '=='}, {'input': '216, 935', 'output': '1151', 'relation': '=='}, {'input': '965, 580', 'output': '1545', 'relation': '=='}, {'input': '897, 735', 'output': '1632', 'relation': '=='}, {'input': '322, 217', 'output': '539', 'relation': '=='}, {'input': '671, 511', 'output': '1182', 'relation': '=='}, {'input': '405, 905', 'output': '1310', 'relation': '=='}, {'input': '936, 658', 'output': '1594', 'relation': '=='}, {'input': '469, 146', 'output': '615', 'relation': '=='}, {'input': '271, 142', 'output': '413', 'relation': '=='}, {'input': '252, 762', 'output': '1014', 'relation': '=='}, {'input': '574, 551', 'output': '1125', 'relation': '=='}, {'input': '269, 764', 'output': '1033', 'relation': '=='}, {'input': '598, 438', 'output': '1036', 'relation': '=='}, {'input': '919, 597', 'output': '1516', 'relation': '=='}, {'input': '408, 370', 'output': '778', 'relation': '=='}, {'input': '224, 141', 'output': '365', 'relation': '=='}, {'input': '521, 505', 'output': '1026', 'relation': '=='}, {'input': '93, 773', 'output': '866', 'relation': '=='}, {'input': '48, 881', 'output': '929', 'relation': '=='}, {'input': '112, 156', 'output': '268', 'relation': '=='}, {'input': '642, 163', 'output': '805', 'relation': '=='}, {'input': '811, 696', 'output': '1507', 'relation': '=='}, {'input': '432, 610', 'output': '1042', 'relation': '=='}, {'input': '65, 394', 'output': '459', 'relation': '=='}, {'input': '390, 610', 'output': '1000', 'relation': '=='}, {'input': '479, 541', 'output': '1020', 'relation': '=='}, {'input': '257, 994', 'output': '1251', 'relation': '=='}, {'input': '566, 881', 'output': '1447', 'relation': '=='}, {'input': '965, 11', 'output': '976', 'relation': '=='}, {'input': '696, 738', 'output': '1434', 'relation': '=='}, {'input': '117, 698', 'output': '815', 'relation': '=='}]
test_case_dic['HumanEval/56'] = [
    {'input': "'<>'", 'output': 'True', 'relation': '=='},
    {'input': "'<<><>>'", 'output': 'True', 'relation': '=='},
    {'input': "'<><><<><>><>'", 'output': 'True', 'relation': '=='},
    {'input': "'<><><<<><><>><>><<><><<>>>'", 'output': 'True', 'relation': '=='},
    {'input': "'<<<><>>>>'", 'output': 'False', 'relation': '=='},
    {'input': "'><<>'", 'output': 'False', 'relation': '=='},
    {'input': "'<'", 'output': 'False', 'relation': '=='},
    {'input': "'<<<<'", 'output': 'False', 'relation': '=='},
    {'input': "'>'", 'output': 'False', 'relation': '=='},
    {'input': "'<<>'", 'output': 'False', 'relation': '=='},
    {'input': "'<><><<><>><>><<>'", 'output': 'False', 'relation': '=='},
    {'input': "'<><><<><>><>>><>'", 'output': 'False', 'relation': '=='}
]
test_case_dic['HumanEval/61'] = [
    {'input': "'()'", 'output': 'True', 'relation': '=='},
    {'input': "'(()())'", 'output': 'True', 'relation': '=='},
    {'input': "'()()(()())()'", 'output': 'True', 'relation': '=='},
    {'input': "'()()((()()())())(()()(()))'", 'output': 'True', 'relation': '=='},
    {'input': "'((()())))'", 'output': 'False', 'relation': '=='},
    {'input': "')(()'", 'output': 'False', 'relation': '=='},
    {'input': "'('", 'output': 'False', 'relation': '=='},
    {'input': "'(((('", 'output': 'False', 'relation': '=='},
    {'input': "')'", 'output': 'False', 'relation': '=='},
    {'input': "'(()'", 'output': 'False', 'relation': '=='},
    {'input': "'()()(()())())(()'", 'output': 'False', 'relation': '=='},
    {'input': "'()()(()())()))()'", 'output': 'False', 'relation': '=='}
]
test_case_dic['HumanEval/72'] = [
    {'input': '[3, 2, 3], 9', 'output': 'True', 'relation': '=='},
    {'input': '[1, 2], 5', 'output': 'False', 'relation': '=='},
    {'input': '[3], 5', 'output': 'True', 'relation': '=='},
    {'input': '[3, 2, 3], 1', 'output': 'False', 'relation': '=='},
    {'input': '[1, 2, 3], 6', 'output': 'False', 'relation': '=='},
    {'input': '[5], 5', 'output': 'True', 'relation': '=='}
]

test_case_dic['HumanEval/76'] = [
    {'input': '16, 2', 'output': 'True', 'relation': '=='},
    {'input': '143214, 16', 'output': 'False', 'relation': '=='},
    {'input': '4, 2', 'output': 'True', 'relation': '=='},
    {'input': '9, 3', 'output': 'True', 'relation': '=='},
    {'input': '16, 4', 'output': 'True', 'relation': '=='},
    {'input': '24, 2', 'output': 'False', 'relation': '=='},
    {'input': '128, 4', 'output': 'False', 'relation': '=='},
    {'input': '12, 6', 'output': 'False', 'relation': '=='}
]
test_case_dic['HumanEval/92'] = [
    {'input': '2, 3, 1', 'output': 'True', 'relation': '=='},
    {'input': '2.5, 2, 3', 'output': 'False', 'relation': '=='},
    {'input': '1.5, 5, 3.5', 'output': 'False', 'relation': '=='},
    {'input': '2, 6, 2', 'output': 'False', 'relation': '=='},
    {'input': '4, 2, 2', 'output': 'True', 'relation': '=='},
    {'input': '2.2, 2.2, 2.2', 'output': 'False', 'relation': '=='},
    {'input': '-4, 6, 2', 'output': 'True', 'relation': '=='},
    {'input': '2, 1, 1', 'output': 'True', 'relation': '=='},
    {'input': '3, 4, 7', 'output': 'True', 'relation': '=='},
    {'input': '3.0, 4, 7', 'output': 'False', 'relation': '=='}
]
test_case_dic['HumanEval/133'] = [
    {'input': '[1, 2, 3]', 'output': '14', 'relation': '=='},
    {'input': '[1.0, 2, 3]', 'output': '14', 'relation': '=='},
    {'input': '[1,3,5,7]', 'output': '84', 'relation': '=='},
    {'input': '[1.4,4.2,0]', 'output': '29', 'relation': '=='},
    {'input': '[-2.4,1,1]', 'output': '6', 'relation': '=='},
    {'input': '[100,1,15,2]', 'output': '10230', 'relation': '=='},
    {'input': '[10000,10000]', 'output': '200000000', 'relation': '=='},
    {'input': '[-1.4,4.6,6.3]', 'output': '75', 'relation': '=='},
    {'input': '[-1.4,17.9,18.9,19.9]', 'output': '1086', 'relation': '=='},
    {'input': '[0]', 'output': '0', 'relation': '=='},
    {'input': '[-1]', 'output': '1', 'relation': '=='},
    {'input': '[-1,1,0]', 'output': '2', 'relation': '=='}
]

test_case_dic['HumanEval/135'] = [
    {'input': '[1,2,4,3,5]', 'output': '3', 'relation': '=='},
    {'input': '[1,2,4,5]', 'output': '-1', 'relation': '=='},
    {'input': '[1,4,2,5,6,7,8,9,10]', 'output': '2', 'relation': '=='},
    {'input': '[4,8,5,7,3]', 'output': '4', 'relation': '=='},
    {'input': '[]', 'output': '-1', 'relation': '=='}
]
for case in test_case_dic['HumanEval/68']:
    case['output'] = case['output'].replace(', "Error"', '')
for case in test_case_dic['HumanEval/88']:
    case['output'] = case['output'].replace(', "Error"', '')
for case in test_case_dic['HumanEval/105']:
    case['output'] = case['output'].replace(', "Error"', '')
for case in test_case_dic['HumanEval/159']:
    case['output'] = case['output'].replace(', "Error"', '')
test_case_dic['HumanEval/109'] = [
    {'input': '[3, 4, 5, 1, 2]', 'output': 'True', 'relation': '=='},
    {'input': '[3, 5, 10, 1, 2]', 'output': 'True', 'relation': '=='},
    {'input': '[4, 3, 1, 2]', 'output': 'False', 'relation': '=='},
    {'input': '[3, 5, 4, 1, 2]', 'output': 'False', 'relation': '=='},
    {'input': '[]', 'output': 'True', 'relation': '=='}
]
test_case_dic['HumanEval/111'] = [
    {'input': "'a b b a'", 'output': "{'a':2,'b': 2}", 'relation': '=='},
    {'input': "'a b c a b'", 'output': "{'a': 2, 'b': 2}", 'relation': '=='},
    {'input': "'a b c d g'", 'output': "{'a': 1, 'b': 1, 'c': 1, 'd': 1, 'g': 1}", 'relation': '=='},
    {'input': "'r t g'", 'output': "{'r': 1,'t': 1,'g': 1}", 'relation': '=='},
    {'input': "'b b b b a'", 'output': "{'b': 4}", 'relation': '=='},
    {'input': "''", 'output': "{}", 'relation': '=='},
    {'input': "'a'", 'output': "{'a': 1}", 'relation': '=='}
]
test_case_dic['HumanEval/113'] = [
    {'input': "['1234567']", 'output': "['the number of odd elements 4n the str4ng 4 of the 4nput.']", 'relation': '=='},
    {'input': "['3','11111111']", 'output': "['the number of odd elements 1n the str1ng 1 of the 1nput.', 'the number of odd elements 8n the str8ng 8 of the 8nput.']", 'relation': '=='},
    {'input': "['271', '137', '314']", 'output': "['the number of odd elements 2n the str2ng 2 of the 2nput.', 'the number of odd elements 3n the str3ng 3 of the 3nput.', 'the number of odd elements 2n the str2ng 2 of the 2nput.']", 'relation': '=='}
]
# statement after assertation false
assertation_comment_list = [64, 65, 66, 71, 76, 77, 79, 78, 80, 84, 89, 91, 92, 93, 94, 95, 97, 99, 107, 114, 115, 122, 126, 132, 133, 139, 140, 144, 151, 154, 157, 160]
for i in assertation_comment_list:
    for case in test_case_dic['HumanEval/%s' % (i)]:
        # if ',' in res['output']:
        #     res['output'] = res['output'].split(',')[0]
        case['output'] = case['output'].split(',')[0]
test_case_dic['HumanEval/65'][3]['output'] = test_case_dic['HumanEval/65'][3]['output'][1:-1]
test_case_dic['HumanEval/65'][4]['output'] = test_case_dic['HumanEval/65'][4]['output'][1:-1]
test_case_dic['HumanEval/79'][3]['output'] = test_case_dic['HumanEval/79'][3]['output'][1:-1]
for case in test_case_dic['HumanEval/84']:
    case['output'] = case['output'][:-1]
quota_list = [89, 93, 140]
for i in quota_list:
    for case in test_case_dic['HumanEval/%s' % (i)]:
        case['output'] = case['output'][1:-1]

test_case_dic['HumanEval/100'] = [
    {'input': '3', 'output': '[3, 5, 7]', 'relation': '=='},
    {'input': '4', 'output': '[4, 6, 8, 10]', 'relation': '=='},
    {'input': '5', 'output': '[5, 7, 9, 11, 13]', 'relation': '=='},
    {'input': '6', 'output': '[6, 8, 10, 12, 14, 16]', 'relation': '=='},
    {'input': '8', 'output': '[8, 10, 12, 14, 16, 18, 20, 22]', 'relation': '=='}
]

test_case_dic['HumanEval/117'] = [
    {'input': '"Mary had a little lamb", 4','output': '["little"]', 'relation': '=='},
    {'input': '"Mary had a little lamb", 3','output': '["Mary", "lamb"]', 'relation': '=='},
    {'input': '"simple white space", 2', 'output': '[]', 'relation': '=='},
    {'input': '"Hello world", 4', 'output': '["world"]', 'relation': '=='},
    {'input': '"Uncle sam", 3', 'output': '["Uncle"]', 'relation': '=='},
    {'input': '"", 4', 'output': '[]', 'relation': '=='},
    {'input': '"a b c d e f", 1', 'output': '["b", "c", "d", "f"]', 'relation': '=='}
]
test_case_dic['HumanEval/148'] = [
    {'input': '"Jupiter", "Neptune"', 'output': '("Saturn", "Uranus")', 'relation': '=='},
    {'input': '"Earth", "Mercury"', 'output': '("Venus",)', 'relation': '=='},
    {'input': '"Mercury", "Uranus"', 'output': '("Venus", "Earth", "Mars", "Jupiter", "Saturn")', 'relation': '=='},
    {'input': '"Neptune", "Venus"', 'output': '("Earth", "Mars", "Jupiter", "Saturn", "Uranus")', 'relation': '=='},
    {'input': '"Earth", "Earth"', 'output': '()', 'relation': '=='},
    {'input': '"Mars", "Earth"', 'output': '()', 'relation': '=='},
    {'input': '"Jupiter", "Makemake"', 'output': '()', 'relation': '=='}
]
test_case_dic['HumanEval/158'] = [
    {'input': '["name", "of", "string"]', 'output': 'string', 'relation': '=='},
    {'input': '["name", "enam", "game"]', 'output': 'enam', 'relation': '=='},
    {'input': '["aaaaaaa", "bb", "cc"]', 'output': 'aaaaaaa', 'relation': '=='},
    {'input': '["abc", "cba"]', 'output': 'abc', 'relation': '=='},
    {'input': '["play", "this", "game", "of","footbott"]', 'output': 'footbott', 'relation': '=='},
    {'input': '["we", "are", "gonna", "rock"]', 'output': 'gonna', 'relation': '=='},
    {'input': '["we", "are", "a", "mad", "nation"]', 'output': 'nation', 'relation': '=='},
    {'input': '["this", "is", "a", "prrk"]', 'output': 'this', 'relation': '=='},
    {'input': '["b"]', 'output': 'b', 'relation': '=='},
    {'input': '["play", "play", "play"]', 'output': 'play', 'relation': '=='}
]
test_case_dic['HumanEval/151'][-1] = {
    'input': '[-99, -97, -95, -93, -91, -89, -87, -85, -83, -81, -79, -77, -75, -73, -71, -69, -67, -65, -63, -61, -59, -57, -55, -53, -51, -49, -47, -45, -43, -41, -39, -37, -35, -33, -31, -29, -27, -25, -23, -21, -19, -17, -15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99]',
    'output': '166650',
    'relation': '=='
}
test_case_dic['HumanEval/152'] = [
    {'input': '[1,2,3,4,5,1],[1,2,3,4,2,-2]', 'output': '[0,0,0,0,3,3]', 'relation': '=='},
    {'input': '[0,0,0,0,0,0],[0,0,0,0,0,0]', 'output': '[0,0,0,0,0,0]', 'relation': '=='},
    {'input': '[1,2,3],[-1,-2,-3]', 'output': '[2,4,6]', 'relation': '=='},
    {'input': '[1,2,3,5],[-1,2,3,4]', 'output': '[2,0,0,1]', 'relation': '=='}
]
test_case_dic['HumanEval/123'] = [
    {'input': '14', 'output': '[1, 5, 7, 11, 13, 17]', 'relation': '=='},
    {'input': '5', 'output': '[1, 5]', 'relation': '=='},
    {'input': '12', 'output': '[1, 3, 5]', 'relation': '=='},
    {'input': '1', 'output': '[1]', 'relation': '=='}
]
test_case_dic['HumanEval/107'] = [
    {'input': '123', 'output': '(8, 13)', 'relation': '=='},
    {'input': '12', 'output': '(4, 6)', 'relation': '=='},
    {'input': '3', 'output': '(1, 2)', 'relation': '=='},
    {'input': '63', 'output': '(6, 8)', 'relation': '=='},
    {'input': '25', 'output': '(5, 6)', 'relation': '=='},
    {'input': '19', 'output': '(4, 6)', 'relation': '=='},
    {'input': '9', 'output': '(4, 5)', 'relation': '=='},
    {'input': '1', 'output': '(0, 1)', 'relation': '=='}
]
test_case_dic['HumanEval/163'] = [
    {'input': '2, 10', 'output': '[2, 4, 6, 8]', 'relation': '=='},
    {'input': '10, 2', 'output': '[2, 4, 6, 8]', 'relation': '=='},
    {'input': '132, 2', 'output': '[2, 4, 6, 8]', 'relation': '=='},
    {'input': '17, 89', 'output': '[]', 'relation': '=='}
]
# format problem
format_problem_list = [71, 96, 101, 105, 108, 111, 112, 117, 125, 148, 152, 149]
for i in format_problem_list:
    for case in test_case_dic['HumanEval/%s' % (i)]:
        case['output'] = str(eval(case['output']))


# def test_solution():
case_status = []
for i in range(len(problem_list)):
    test_cases = test_case_dic[problem_list[i]['task_id']]
    demo_file = 'demo.py'
    with open(demo_file, 'w', encoding='utf-8') as f:
        f.write(problem_list[i]['prompt'] + problem_list[i]['canonical_solution'])
    call_demo_file = 'call_demo.py'
    unpassed_test_case = []
    for j in range(len(test_cases)):
        if test_cases[j]['relation'] == '==':
            with open('./call_demo.py', 'w') as f:
                f.write('from %s import %s\nprint(%s(%s))' % (
                    demo_file.split('.')[0],
                    problem_list[i]['entry_point'],
                    problem_list[i]['entry_point'],
                    test_cases[j]['input']
                ))
            try:
                output = subprocess.run(["python", call_demo_file], capture_output=True, text=True, timeout=3)

            except subprocess.TimeoutExpired as e:
                print(e, flush=True)
                unpassed_test_case.append([j, 'Timeout'])
                continue
            except Exception as e:
                print(e, flush=True)
                unpassed_test_case.append([j, 'Exception'])
                continue
            if test_cases[j]['output'].strip() != output.stdout.strip():
                unpassed_test_case.append([j, 'false'])
            else:
                unpassed_test_case.append([j, 'True'])
        else:
            if '$input$' in test_cases[j]['relation'] or '$demo$' in test_cases[j]['relation']:
                with open('./call_demo.py', 'w') as f:
                    f.write('from %s import %s\n%s' % (
                        demo_file.split('.')[0],
                        problem_list[i]['entry_point'],
                        test_cases[j]['relation'].replace('$input$', str(test_cases[j]['input'])).replace('$demo$', demo_file.split('.')[0])
                    ))
            else:
                with open('./call_demo.py', 'w') as f:
                    f.write('from %s import %s\nprint(%s)' % (demo_file.split('.')[0],
                        problem_list[i]['entry_point'],
                        test_cases[j]['relation'].replace('candidate', problem_list[i]['entry_point'])))
            try:
                output = subprocess.run(["python", call_demo_file], capture_output=True, text=True, timeout=3)
            except subprocess.TimeoutExpired as e:
                print(e, flush=True)
                unpassed_test_case.append([j, 'Timeout'])
                continue
            except Exception as e:
                print(e, flush=True)
                unpassed_test_case.append([j, 'Exception'])
                continue
            if output.stdout.strip() != 'True':
                unpassed_test_case.append([j, 'false'])
            else:
                unpassed_test_case.append([j, 'True'])
    if len(set([i[1] for i in unpassed_test_case])) == 1 and unpassed_test_case[0][1] == 'True':
        # print('ALL TRUE')
        case_status.append(['ALL TRUE'])
    else:
        case_status.append(unpassed_test_case)
    # print(unpassed_test_case)

    # test_cases = test_case_dic[problem_list[i]['task_id']]
    # with open('./demo.py', 'w', encoding='utf-8') as f:
    #     f.write(problem_list[i]['prompt'] + problem_list[i]['canonical_solution'])
    # call_demo_file = 'call_demo.py'
    # unpassed_test_case = []
    # for j in range(len(test_cases)):
    #     if test_cases[j]['relation'] == '==':
    #         with open('./call_demo.py', 'w') as f:
    #             f.write('from demo import %s\nprint(%s(%s))' % (
    #             problem_list[i]['entry_point'], problem_list[i]['entry_point'], test_cases[j]['input']))
    #         try:
    #             output = subprocess.run(["python", call_demo_file], capture_output=True, text=True, timeout=3)
    #
    #         except subprocess.TimeoutExpired as e:
    #             print(e, flush=True)
    #             unpassed_test_case.append([j,'Timeout'])
    #             continue
    #         except Exception as e:
    #             print(e, flush=True)
    #             unpassed_test_case.append([j,'Exception'])
    #             continue
    #         if test_cases[j]['output'].strip() != output.stdout.strip():
    #             unpassed_test_case.append([j, 'false'])
    #     else:
    #         with open('./call_demo.py', 'w') as f:
    #             f.write('from demo import %s\nprint(%s)' % (
    #             problem_list[i]['entry_point'], test_cases[j]['relation'].replace('candidate', problem_list[i]['entry_point'])))
    #         try:
    #             output = subprocess.run(["python", call_demo_file], capture_output=True, text=True, timeout=3)
    #         except subprocess.TimeoutExpired as e:
    #             print(e, flush=True)
    #             unpassed_test_case.append([j,'Timeout'])
    #             continue
    #         except Exception as e:
    #             print(e, flush=True)
    #             unpassed_test_case.append([j,'Exception'])
    #             continue
    #         if output.stdout.strip() != 'True':
    #             unpassed_test_case.append([j, 'false'])
    # case_status.append(unpassed_test_case)

# test_case = []
# for i in range(len(input)):
#     res = {'input': input[i], 'output': output[i], 'relation': '=='}
#     test_case.append(res)

# reconstruct the dataset
if not os.path.exists('HumanEval/HumanEval_new.jsonl'):
    with open('HumanEval/HumanEval_new.json', 'w') as f:
        f.write('')


for problem in problem_list:
    res = {
        'name': problem['task_id'],
        'entry_point': problem['entry_point'],
        'prompt': problem['prompt'],
        'solution': problem['prompt'] + problem['canonical_solution'],
        'test_case': test_case_dic[problem['task_id']]
    }
    json_str = json.dumps(res)
    with open('HumanEval/HumanEval_new.jsonl', 'a') as f:
        f.write(json_str + '\n')
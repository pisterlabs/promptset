import numpy as np
import os
import openai
import time
import signal
import json
from basic_framework.utils import regularize
from basic_framework.core_testing import Tester

openai.api_key = "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"


def repair(path):
    if not 'data_des' in path:
        fixed_id = json.load(open("fixed_codex_id.json"))
    else:
        fixed_id = json.load(open("fixed_codex_id_des.json"))
    # keep same with ChatGPT
    descriptions = [
        'This function takes in a value “x” and a sorted sequence “seq”, and returns the position that “x” should go to such that the sequence remains sorted. Otherwise, return the length of the sequence.',
        'Given a month and a list of possible birthdays, these functions check if there is only one possible birthday with that month and unique day. Three different functions are implemented: unique_day, unique_month and contains_unique_day.',
        'This function takes in a list and returns a new list with all repeated occurrences of any element removed and the order of the original elements kept.',
        'Given a list of people, this function sorts the people and returns a list in an order such that the older people are at the front of the list.',
        'This function top_k accepts a list of integers as the input and returns the greatest k number of values as a list, with its elements sorted in descending order.']

    # signal.signal(signal.SIGALRM, handler)
    questions = ['question_1', 'question_2', 'question_3', 'question_4', 'question_5']
    for i in range(len(questions)):
        q = questions[i]
        des = descriptions[i]
        print(q)

        ids = fixed_id[q]
        cnt = 0
        response_time = []
        path_question = path + q
        path_fixed_code = os.path.join(path_question, 'code/fixed_codex')
        if not os.path.exists(path_fixed_code):
            os.makedirs(path_fixed_code)

        path_buggy_code = os.path.join(path_question, 'code/wrong')
        assignments_wrong = os.listdir(path_buggy_code)
        for assign in assignments_wrong:
            cnt += 1
            buggy_file_name = assign
            if buggy_file_name.startswith('.'):
                continue
            # fixed_file_name = buggy_file_name.replace('wrong', 'fixed')
            # if os.path.exists(os.path.join(path_fixed_code, fixed_file_name)):
            #     continue
            id = assign.split('_')[2]
            fixed_file_name = buggy_file_name.replace('wrong', 'fixed5')
            # if id in ids or os.path.exists(os.path.join(path_fixed_code, fixed_file_name)):
            if os.path.exists(os.path.join(path_fixed_code, fixed_file_name)):
                continue

            path_wrong_assign = os.path.join(path_buggy_code, assign)
            buggy_version_code = open(path_wrong_assign).read().strip()
            # prompt = "##### There are one or more bugs in the below code. Can you please fix them? Reply me only with the fixed code. Do not include any natural language words, notes or explanations in your answer.\n \n### Buggy Python\n" + buggy_version_code + "\n### Fixed Python"
            if q == 'question_2':
                prompt = "##### Fix bugs in the below functions\n \n### Buggy Python\n" + buggy_version_code + "\n### Fixed Python"
                # prompt = "##### Fix bugs in the below functions\n" +des+ "\n### Buggy Python\n" + buggy_version_code + "\n### Fixed Python"
            else:
                prompt = "##### Fix bugs in the below function\n \n### Buggy Python\n" + buggy_version_code + "\n### Fixed Python"
                # prompt = "##### Fix bugs in the below function\n" +des+ "\n### Buggy Python\n" + buggy_version_code + "\n### Fixed Python"
            begin = time.time()

            try:

                # OpenAI API
                response = openai.Completion.create(
                    model="code-davinci-002",
                    prompt=prompt,
                    # temperature=0,
                    max_tokens=1024,
                    # top_p=1.0,
                    # frequency_penalty=0.0,
                    # presence_penalty=0.0,
                    stop=["###"]
                )

                answer = response.choices[0].text.strip().strip(',').strip('.')
                # remove natural language
                answer_list = answer.strip().split('\n')
                while not answer_list[0].startswith("def "):
                    answer_list.pop(0)
                pure_code = '\n'.join(answer_list)

                with open(os.path.join(path_fixed_code, fixed_file_name), 'w+') as file:
                    file.write(pure_code)
            except Exception as e:
                # raise
                print(e.__str__())
                if 'unavailable' in e.__str__():
                    print('waiting 10 min to request again ...')
                    time.sleep(60 * 10)
                continue
            end = time.time()
            cost = end - begin
            if cost <= 5:
                time.sleep(5 - cost)
            print('{}: {}'.format(cnt, buggy_file_name))
        response_time_average = np.array(response_time).mean()
        print('Response time average:')
        print('{}: {}'.format(q, response_time_average))


def validate(path, metric):
    all_result = []
    fixed_id = {}
    check = []
    for q in ['question_1', 'question_2', 'question_3', 'question_4', 'question_5']:
        print(q)
        fixed_id[q] = []
        question_path = path + q
        t = Tester(question_path)
        path_fixed_code = os.path.join(question_path, 'code/fixed_codex')
        assignments_fixed = os.listdir(path_fixed_code)
        cnt, correct, wrong, error = 0, 0, 0, 0
        for assign in assignments_fixed:
            if assign.startswith('.'):
                continue
            cnt += 1
            file_name = assign
            # print(file_name)
            # if not file_name.startswith('fixed_'):# TOP-1
            #     continue
            # if not (file_name.startswith('fixed_') or file_name.startswith('fixed2_') or file_name.startswith('fixed3_') or file_name.startswith('fixed4_')):# TOP-N
            #     continue
            path_fixed_assign = os.path.join(path_fixed_code, assign)
            # print(file_name, end=' ')
            if file_name == 'fixed4_3_075.pure.py' or file_name == 'fixed5_1_219.pure.py':
                print('{}: incorrect!'.format(file_name))
                wrong += 1
                continue
            with open(path_fixed_assign, "r") as f:
                # file_name = path_fixed_assign.split("/")[-1]
                try:
                    corr_code = regularize(f.read())
                    tr = t.tv_code(corr_code)
                except Exception as e:
                    print("{}: error!".format(file_name))
                    error += 1
                    check.append(file_name)
                    continue
                    # raise e
                if t.is_pass(tr):
                    # corr_code_map[file_name] = corr_code
                    print('{}: correct!'.format(file_name))
                    correct += 1
                    # q_id = file_name.split('_')[1]
                    assign_id = file_name.split('_')[2]
                    if assign_id not in fixed_id[q]:
                        fixed_id[q].append(assign_id)
                else:
                    print('{}: incorrect!'.format(file_name))
                    wrong += 1
                    # print(tr)
                    # print(path_fixed_assign)
                    # shutil.move(corr_code_path, pseudo_corr_dir_path)
        all_result.append([cnt, correct, wrong, error, correct/cnt])
    # print('lets check: {}'.format(sorted(check)))
    if not 'data_des' in path:
        json.dump(fixed_id, open('fixed_codex_id.json', 'w+'))
    else:
        json.dump(fixed_id, open('fixed_codex_id_des.json', 'w+'))

    # if metric == 'AVG5':
    print('AVG-5')
    print("cnt, correct, wrong, error, fix rate")
    print(all_result[0],)
    print(all_result[1])
    print(all_result[2])
    print(all_result[3])
    print(all_result[4])
    all_cnt = all_result[0][0] + all_result[1][0] + all_result[2][0] + all_result[3][0] + all_result[4][0]
    all_correctness = all_result[0][1] + all_result[1][1] + all_result[2][1] + all_result[3][1] + all_result[4][1]
    print('################')
    print('All: fix rate: {}'.format(all_correctness/all_cnt))

def calculate(path):
    print('TOP-5')
    all_cnt = 1783
    fix_cnt = 0
    if not 'data_des' in path:
        with open('fixed_codex_id.json', 'r+') as f:
            dict = json.load(f)
    else:
        with open('fixed_codex_id_des.json', 'r+') as f:
            dict = json.load(f)
    for k,v in dict.items():
        fix_cnt_ques = len(set(v))
        fix_cnt += fix_cnt_ques
        if k == 'question_1':
            cnt = 575
        elif k == 'question_2':
            cnt = 435
        elif k == 'question_3':
            cnt = 308
        elif k == 'question_4':
            cnt = 357
        elif k == 'question_5':
            cnt = 108
        print('Question: {}, Fix :{}, Fix rate: {}'.format(k, fix_cnt_ques, round(fix_cnt_ques/cnt, 4)))
    print('################')
    print('All: fix_cnt: {}, fix_rate: {}'.format(fix_cnt, round(fix_cnt/all_cnt, 4)))


if __name__ == '__main__':

    path = '/Users/haoye.tian/Documents/University/project/refactory/data_des/'
    # repair(path)
    validate(path, 'AVG5')
    # calculate()


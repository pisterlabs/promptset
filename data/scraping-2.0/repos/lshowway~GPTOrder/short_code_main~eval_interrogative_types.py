import json
import numpy as np
import nltk.translate.bleu_score as bleu
from random import random, shuffle
from time import sleep

import openai
openai.api_key="sk-XXX"
model_engien = "text-davinci-003"


def different_type(q_w='when', this_id=-1, this_count=20):
    with open('D:/phd5/KR/natural_questions.jsonl', 'r', encoding='utf-8') as f:
        all_questions = []
        for line in f:
            json_obj = json.loads(line)
            all_questions.append(json_obj)
        all_questions = all_questions[1000: -1000]

    with open('D:/phd5/KR/different_interrogative/natural_questions_answers_%s.jsonl'%q_w, 'a', encoding='utf-8') as fw:
        count = 0
        for json_obj in all_questions:
            id = json_obj['id']
            if id <= this_id:
                continue
            length = json_obj['length']
            question = json_obj['question']
            if q_w == 'be':
                t = question.lower().startswith('is') or question.lower().startswith('are') or question.lower().startswith('was') or question.lower().startswith('were')
            elif q_w == 'do':
                t = question.lower().startswith('do') or question.lower().startswith('does') or question.lower().startswith('did')
            elif q_w == 'modal':
                t = question.lower().startswith('can') or \
                    question.lower().startswith('could') or \
                    question.lower().startswith('may') or \
                    question.lower().startswith('might') or \
                    question.lower().startswith('must') or \
                    question.lower().startswith('shall') or \
                    question.lower().startswith('should') or \
                    question.lower().startswith('will') or \
                    question.lower().startswith('would')
            elif q_w == 'how':
                t = question.lower().startswith('how')
            elif q_w == 'when':
                t = question.lower().startswith('when')
            elif q_w == 'where':
                t = question.lower().startswith('where')
            elif q_w == 'which':
                t = question.lower().startswith('which')
            elif q_w == 'who':
                t = question.lower().startswith('who') or question.lower().startswith('whose')
            elif q_w == 'why':
                t = question.lower().startswith('why')
            elif q_w == 'what':
                t = question.lower().startswith('what')
            else:
                t = False
            if t:
                count += 1
                if count <= this_count:
                    q_list = question.split()
                    answer = openai.Completion.create(engine=model_engien, prompt=question, max_tokens=1024, n=1, stop=None, temperature=0.5).choices[0].text

                    # a. exhange the first and last word
                    a_question = q_list[-1] + ' ' + ' '.join(q_list[1:-1]) + ' ' + q_list[0]
                    a_answer = openai.Completion.create(engine=model_engien, prompt=a_question, max_tokens=1024, n=1, stop=None, temperature=0.5).choices[0].text

                    # b. exchange adjacent words
                    new_i = []
                    for i in range(len(q_list) // 2):
                        t = [2*i+1, 2*i]
                        new_i.extend(t)
                    if len(q_list) % 2 == 1:
                        new_i.append(len(q_list) - 1)
                    b_question = ' '.join([q_list[i] for i in new_i])
                    b_answer = \
                    openai.Completion.create(engine=model_engien, prompt=b_question, max_tokens=1024, n=1, stop=None,
                                             temperature=0.5).choices[0].text

                    # c. fix the first and last word, shuffle others
                    t2 = q_list[1:-1]
                    shuffle(t2)
                    c_question = q_list[0] + ' ' + ' '.join(t2) + ' ' + q_list[-1]
                    c_answer = \
                    openai.Completion.create(engine=model_engien, prompt=c_question, max_tokens=1024, n=1, stop=None,
                                             temperature=0.5).choices[0].text

                    # I. shuffle in trigram
                    new_j = []
                    for i in range(len(q_list) // 3):
                        j = [3*i, 3*i+1, 3*i+2]
                        shuffle(j)
                        new_j.extend(j)
                    if len(q_list) % 3 == 1:
                        new_j.append(len(q_list) - 1)
                    elif len(q_list) % 3 == 2:
                        new_j.extend([len(q_list) - 2, len(q_list) - 1])
                    I_question = ' '.join([q_list[i] for i in new_j])
                    I_answer = \
                    openai.Completion.create(engine=model_engien, prompt=I_question, max_tokens=1024, n=1, stop=None,
                                             temperature=0.5).choices[0].text

                    # II. reverse the order of words
                    II_question = ' '.join(q_list[::-1])
                    II_answer = \
                    openai.Completion.create(engine=model_engien, prompt=II_question, max_tokens=1024, n=1, stop=None,
                                             temperature=0.5).choices[0].text

                    print(question)
                    print(a_question)
                    print(b_question)
                    print(c_question)
                    print(I_question)
                    print(II_question)
                    print('------------------')

                    json.dump({"id": id, "length": length, "prompt": question, "answer": answer,
                               "exchange": a_answer, "adjecient": b_answer, "fix": c_answer,
                               "trigram": I_answer, "reverse": II_answer}, fw)
                    fw.write('\n')


def compute_blue(file):
    # python short_code_main to compute BLUE score
    results_list = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            json_obj = json.loads(line)
            id = json_obj['id']
            length = json_obj['length']
            answer = json_obj['answer']
            a_answer = json_obj['exchange']
            b_answer = json_obj['adjecient']
            c_answer = json_obj['fix']
            I_answer = json_obj['trigram']
            II_answer = json_obj['reverse']

            a_score = bleu.sentence_bleu([answer.split()], a_answer.split(), weights=(1, 0, 0, 0))
            b_score = bleu.sentence_bleu([answer.split()], b_answer.split(), weights=(1, 0, 0, 0))
            c_score = bleu.sentence_bleu([answer.split()], c_answer.split(), weights=(1, 0, 0, 0))
            I_score = bleu.sentence_bleu([answer.split()], I_answer.split(), weights=(1, 0, 0, 0))
            II_score = bleu.sentence_bleu([answer.split()], II_answer.split(),  weights=(1, 0, 0, 0))

            results_list.append([round(a_score, 4), round(b_score, 4), round(c_score, 4), round(I_score, 4), round(II_score, 4)])

            # print(id, length, round(a_score, 4), round(b_score, 4), round(c_score, 4), round(I_score, 4), round(II_score, 4))

    results_list = np.array(results_list)

    # print('exchange first & last word', 'adjacent words', 'fix first & last word', 'shuffle in trigram', 'reverse order')
    mean_short_sentences = np.mean(results_list, axis=0)
    # print(mean_short_sentences)
    return mean_short_sentences


if __name__ == "__main__":
    # q_w = 'why'
    # different_type(q_w, this_id=2266, this_count=20 - 19)
    # compute_blue('D:/phd5/KR/different_interrogative/natural_questions_answers_%s.jsonl'%q_w)

    for q_w in ['be', 'do', 'modal', 'who', 'what', 'which', 'where', 'when', 'how', 'why']:
        t = compute_blue('D:/phd5/KR/different_interrogative/natural_questions_answers_%s.jsonl'%q_w)
        print(q_w, round(np.mean(t), 4))


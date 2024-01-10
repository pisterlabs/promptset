import json
from random import random, shuffle
from time import sleep


def different_models(model_engien):
    import openai
    openai.api_key="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"
    # model_engien = "chatgpt"
    # model_engien = "text-davinci-003"

    with open('D:/phd5/KR/different_models/natural_questions_answers_%s.jsonl'%model_engien, 'a', encoding='utf-8') as fw:
        with open('D:/phd5/KR/different_length/natural_questions_length.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                json_obj = json.loads(line)
                id = json_obj['id']

                if id <= 12:
                    continue

                length = json_obj['length']
                question = json_obj['question']
                q_list = question.split()
                answer = openai.Completion.create(engine=model_engien, prompt=question, max_tokens=1024, n=1, stop=None, temperature=0.5).choices[0].text

                print(answer)
                print('------------------')

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

                sleep(10)

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


def compute_blue(model_engien):
    # python short_code_main to compute BLUE score
    import nltk.translate.bleu_score as bleu
    import json
    results_list = []
    with open('D:/phd5/KR/different_models/natural_questions_answers_%s.jsonl'%model_engien, 'r', encoding='utf-8') as f:
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

            print(id, length, round(a_score, 4), round(b_score, 4), round(c_score, 4), round(I_score, 4), round(II_score, 4))

    import numpy as np
    results_list = np.array(results_list)
    mean_short_sentences = np.mean(results_list, axis=0)
    print(mean_short_sentences)


if __name__ == '__main__':
    # GPT 3.5 series, short_code_main-davinci-002 is too slow
    # model_engine = ['short_code_main-davinci-002', 'text-davinci-002', 'text-davinci-003', 'gpt-3.5-turbo', 'text-curie-001', 'text-babbage-001']
    model_engine = ['text-babbage-001']
    for m in model_engine:
        different_models(m)
        compute_blue(m)


    # GPT-3 series
    # model_engine = ['davinci', 'short_code_main-davinci-001', 'short_code_main-cushman-001', 'instruct-davinci-beta', 'text-davinci-001']

"""
Write a pipeline
first step: read the data, including WinoGrand, SST2, GSM8K, TruthfulQA
second step: process the data, for classification and generation, input and label
third step: closed model, only api, chat, claude, and wenxinyiyan
fourth step: prompt, for different task, different models
fifth step: evaluation. metrics.

"""
import json
import random
import re

random.seed(42)

import pandas as pd


def load_XWinoGrade(filename="../data/xwinograd/en.csv", samples_num=100):
    data = pd.read_csv(filename, encoding='utf-8')
    data = data[:samples_num]
    return data


def load_WiQueen(filename="../", samples_num=100):
    data = pd.read_csv(filename, encoding='utf-8')
    data = data[:samples_num]
    return data


def load_MGSM(filename="../mgsm/mgsm_zh.csv", samples_num=100):
    data = pd.read_csv(filename, encoding='utf-8')
    data = data[:samples_num]
    return data


def load_TruthfulQA(filename="../data/TruthfulQA.csv", samples_num=100):
    data = pd.read_csv(filename, encoding='utf-8')
    data = data[: samples_num]
    return data


def upload_TruthfuleQA_finetune_file():
    import openai
    from generation_04 import Chat_API
    openai.api_key = Chat_API

    # r1 = openai.File.create(
    #     file=open("../data/finetune_info.jsonl", "rb"),
    #     purpose='fine-tune'
    # )
    #
    # r2 = openai.File.create(
    #     file=open("../data/finetune_truth.jsonl", "rb"),
    #     purpose='fine-tune'
    # )
    # print(r1)
    # print(r2)

    # openai.File.delete("file-v5d3GYmLGOg69KQNjWmFwXdU")
    # openai.File.delete("file-ufnrfqApLJ6C3gEdIrZalWNL")

    # print(openai.File.retrieve("file-HkPrEuaVruvJOuxHkQykdD4M"))
    # content = openai.File.download("file-apqyboOwCOaP3GVbx5r4P2X9") # info
    # t = content
    # openai.Model.delete("curie:ft-De4Sh25jjQuQ4SgaX0g6GXDI")
    # openai.Model.delete("curie:ft-Ow0oRgeoIfuLj5ZXhsYm8gVp")

    # print(openai.File.list())
    print(openai.FineTune.list())


def load_TruthfulQA_finetune(filename="../data/finetune_info-all.jsonl", w_file="../data/finetune_info.jsonl"):
    with open(filename, encoding='utf-8') as f:
        yes_list, no_list = [], []
        for line in f:
            line = json.loads(line)
            if line['completion'] == ' yes':
                yes_list.append(line)
            if line['completion'] == ' no':
                no_list.append(line)
    all_list = yes_list[: 500] + no_list[: 500]
    import random
    random.shuffle(all_list)
    with open(w_file, 'w', encoding='utf-8') as f:
        for line in all_list:
            json.dump(line, f)
            f.write('\n')


def altering_order_zh(r_file, samples_num=100, w_file=None, altering='ex_random_two'):
    data = pd.read_csv(r_file, encoding='utf-8')
    data = data[:samples_num]

    w_file = w_file.format(altering)

    for index, row in data.iterrows():
        question = row['Question']
        temp = re.compile("([\u4e00-\u9fa5])")
        question = temp.sub(r"\1"+' ' , question)
        question = question.split(' ')
        # exhange random two words in q_list
        if altering == 'ex_random_two':
            i1 = random.randint(0, len(question) - 1)
            i2 = random.randint(0, len(question) - 1)
            t = question[i1]
            question[i1] = question[i2]
            question[i2] = t
            question = ''.join(question)
        elif altering == 'ex_adjacent':
            new_i = []
            for i in range(len(question) // 2):
                t = [2 * i + 1, 2 * i]
                new_i.extend(t)
            if len(question) % 2 == 1:
                new_i.append(len(question) - 1)
            question = ''.join([question[i] for i in new_i])
        elif altering == 'rotate_two_part':
            anchor_index = len(question) // 2

            left_part = question[:anchor_index]
            right_part = question[-anchor_index:]
            question = right_part  + left_part + question[anchor_index:-anchor_index]
            question = ''.join(question)
        data.loc[index, 'Question'] = question
    data.to_csv(w_file, index=False, encoding='utf-8')


def altering_order(r_file, samples_num=100, w_file=None, altering='ex_random_two'):
    data = pd.read_csv(r_file, encoding='utf-8')
    data = data[:samples_num]

    w_file = w_file.format(altering)

    for index, row in data.iterrows():
        question = row['Question'].split()
        # exhange random two words in q_list
        if altering == 'ex_random_two':
            i1 = random.randint(0, len(question) - 1)
            i2 = random.randint(0, len(question) - 1)
            t = question[i1]
            question[i1] = question[i2]
            question[i2] = t
            question = ' '.join(question)
        elif altering == 'ex_adjacent':
            new_i = []
            for i in range(len(question) // 2):
                t = [2 * i + 1, 2 * i]
                new_i.extend(t)
            if len(question) % 2 == 1:
                new_i.append(len(question) - 1)
            question = ' '.join([question[i] for i in new_i])
        elif altering == 'rotate_two_part':
            anchor_index = len(question) // 2

            left_part = question[:anchor_index]
            right_part = question[-anchor_index:]
            question = right_part  + left_part + question[anchor_index:-anchor_index]
            question = ' '.join(question)
        data.loc[index, 'Question'] = question
        data.loc[index, 'Question'] = question
    data.to_csv(w_file, index=False, encoding='utf-8')


def altering_order_wiqueen(r_file, samples_num=100, w_file=None, altering='ex_random_two'):
    data = pd.read_csv(r_file, encoding='utf-8')
    data = data[:samples_num]
    w_file = w_file.format(altering)

    def altering_func(question, altering):
        question = question.split()
        # exhange random two words in q_list
        if altering == 'ex_random_two':
            i1 = random.randint(0, len(question) - 1)
            i2 = random.randint(0, len(question) - 1)
            t = question[i1]
            question[i1] = question[i2]
            question[i2] = t
            question = ' '.join(question)
        elif altering == 'ex_adjacent':
            new_i = []
            for i in range(len(question) // 2):
                t = [2 * i + 1, 2 * i]
                new_i.extend(t)
            if len(question) % 2 == 1:
                new_i.append(len(question) - 1)
            question = ' '.join([question[i] for i in new_i])
        elif altering == 'rotate_two_part':
            anchor_index = len(question) // 2

            left_part = question[:anchor_index]
            right_part = question[-anchor_index:]
            question = right_part + left_part + question[anchor_index:-anchor_index]
            question = ' '.join(question)
        return question

    for index, row in data.iterrows():
        relation = row['relation']
        e1 = row['e1']
        e2 = row['e2']
        e3 = row['e3']
        e4_candidates = row['e4_candidates']

        new_e1 = altering_func(e1, altering=altering)
        new_e2 = altering_func(e2, altering=altering)
        new_e3 = altering_func(e3, altering=altering)

        data.loc[index, 'e1'] = new_e1
        data.loc[index, 'e2'] = new_e2
        data.loc[index, 'e3'] = new_e3
    data.to_csv(w_file, index=False, encoding='utf-8')


if __name__ == "__main__":
    # load_TruthfulQA_finetune()
    # upload_TruthfuleQA_finetune_file()

    # altering_order('../data/TruthfulQA/TruthfulQA.csv', w_file='../data/TruthfulQA/TruthfulQA_{}.csv', altering='ex_random_two')
    # altering_order('../data/TruthfulQA/TruthfulQA.csv', w_file='../data/TruthfulQA/TruthfulQA_{}.csv', altering='ex_adjacent')
    # altering_order('../data/TruthfulQA/TruthfulQA.csv', w_file='../data/TruthfulQA/TruthfulQA_{}.csv', altering='rotate_two_part')

    # altering_order('../data/mgsm/mgsm_en.csv', w_file='../data/mgsm/mgsm_en_{}.csv', altering='ex_random_two')
    # altering_order('../data/mgsm/mgsm_en.csv', w_file='../data/mgsm/mgsm_en_{}.csv', altering='ex_adjacent')
    # altering_order('../data/mgsm/mgsm_en.csv', w_file='../data/mgsm/mgsm_en_{}.csv', altering='rotate_two_part')


    # r_file = "../data/xwinograd/zh.jsonl"
    # w_file = "../data/xwinograd/zh.csv"

    # data = pd.read_json(r_file, lines=True)
    # data.to_csv(w_file, index=False, encoding='utf-8')

    # altering_order_zh('../data/mgsm/mgsm_zh.csv', w_file='../data/mgsm/mgsm_zh_{}.csv', altering='ex_random_two')
    # altering_order_zh('../data/mgsm/mgsm_zh.csv', w_file='../data/mgsm/mgsm_zh_{}.csv', altering='ex_adjacent')
    # altering_order_zh('../data/mgsm/mgsm_zh.csv', w_file='../data/mgsm/mgsm_zh_{}.csv', altering='rotate_two_part')

    # altering_order('../data/mgsm/mgsm_fr.csv', w_file='../data/mgsm/mgsm_fr_{}.csv', altering='ex_random_two')
    # altering_order('../data/mgsm/mgsm_fr.csv', w_file='../data/mgsm/mgsm_fr_{}.csv', altering='ex_adjacent')
    # altering_order('../data/mgsm/mgsm_fr.csv', w_file='../data/mgsm/mgsm_fr_{}.csv', altering='rotate_two_part')

    # altering_order_zh('../data/XWinoGrande/zh.csv', w_file='../data/XWinoGrande/xwinogrande_zh_{}.csv', altering='ex_random_two')
    # altering_order_zh('../data/XWinoGrande/zh.csv', w_file='../data/XWinoGrande/xwinogrande_zh_{}.csv', altering='ex_adjacent')
    # altering_order_zh('../data/XWinoGrande/zh.csv', w_file='../data/XWinoGrande/xwinogrande_zh_{}.csv', altering='rotate_two_part')

    altering_order('../data/XWinoGrande/xwinogrande_fr.csv', w_file='../data/XWinoGrande/xwinogrande_fr_{}.csv', altering='ex_random_two')
    altering_order('../data/XWinoGrande/xwinogrande_fr.csv', w_file='../data/XWinoGrande/xwinogrande_fr_{}.csv', altering='ex_adjacent')
    altering_order('../data/XWinoGrande/xwinogrande_fr.csv', w_file='../data/XWinoGrande/xwinogrande_fr_{}.csv', altering='rotate_two_part')

    # altering_order_wiqueen('../data/WiQueen/analogy_unique_en.csv', w_file='../data/WiQueen/analogy_unique_en_{}.csv', altering='ex_random_two')
    # altering_order_wiqueen('../data/WiQueen/analogy_unique_en.csv', w_file='../data/WiQueen/analogy_unique_en_{}.csv', altering='ex_adjacent')
    # altering_order_wiqueen('../data/WiQueen/analogy_unique_en.csv', w_file='../data/WiQueen/analogy_unique_en_{}.csv', altering='rotate_two_part')

    # altering_order_wiqueen('../data/WiQueen/analogy_unique_fr.csv', w_file='../data/WiQueen/analogy_unique_fr_{}.csv', altering='ex_random_two')
    # altering_order_wiqueen('../data/WiQueen/analogy_unique_fr.csv', w_file='../data/WiQueen/analogy_unique_fr_{}.csv', altering='ex_adjacent')
    # altering_order_wiqueen('../data/WiQueen/analogy_unique_fr.csv', w_file='../data/WiQueen/analogy_unique_fr_{}.csv', altering='rotate_two_part')
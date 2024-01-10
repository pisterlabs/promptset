import csv
import json
import time
import random
random.seed(666)

import pickle
from tqdm import tqdm
import os
import openai
openai.api_key = "YOUR_OPENAI_KEY" # input your openai api_key here


def query_gpt3(sentence):
    response = openai.Completion.create(
        model="code-davinci-002",
        prompt="sentence:\nwhen 's the last time the the organiztion leadership has a person named Terry Collins won the world series\ndecompose:\nFACT1: ENTITY1 is an organization.\nFACT2: ENTITY2 is the leadership of ENTITY1.\nFACT3: ENTITY3 is Terry Collins.\nFACT4: ENTITY3 is a person.\nFACT5: ENTITY2 has ENTITY3.\nFACT6: ENTITY1 is the winner of the world series.\nQuestion: When is the last time of FACT6?\n\nsentence:\nwhat was the person education institution is Detroit Business Institute best known for\ndecompose:\nFACT1: ENTITY1 is Detroit Business Institute.\nFACT2: ENTITY2 is a person.\nFACT3: ENTITY2's education institution is ENTITY1.\nFACT4: ENTITY2 is the best known for ENTITY3.\nQuestion: What is ENTITY3?\n\nsentence:\nwhat type of religion did massachusetts have and the religion organiztion is Army of the Lord\ndecompose:\nFACT1: ENTITY1 is a type of religion.\nFACT2: ENTITY2 is Massachusetts.\nFACT3: ENTITY2 has ENTITY1.\nFACT4: ENTITY3 is Army of the Lord.\nFACT5: ENTITY3 is an organization of ENTITY1.\nQuestion: What is ENTITY1?\n\nsentence:\nwhat are some of the religions in australia and the relgion sacred site is Kushinagar\ndecompose:\nFACT1: ENTITY1 is a religion.\nFACT2: ENTITY2 is Australia.\nFACT3: ENTITY1 is in ENTITY2.\nFACT4: ENTITY3 is Kushinagar.\nFACT5: ENTITY3 is a sacred site of ENTITY1.\nQuestion: What are some of ENTITY1?\n\nsentence:\nwho is the prime minister of the country that contains Wellington Region now\ndecompose:\nFACT1: ENTITY1 is a country.\nFACT2: ENTITY1 contains Wellington Region.\nFACT3: ENTITY2 is the prime minister of ENTITY1 now.\nQuestion: Who is ENTITY2?\n\nsentence:\nwhat are the important holidays of the religion types of places of worship include Masjid Hamza, Valley Stream\ndecompose:\nFACT1: ENTITY1 is a religion.\nFACT2: ENTITY2 is a type of places of worship.\nFACT3: ENTITY2 includes Masjid Hamza, Valley Stream.\nFACT4: ENTITY1 has ENTITY2.\nFACT5: ENTITY3 are important holidays of ENTITY1.\nQuestion: What are ENTITY3?\n\nsentence:\nwhere is the child organization is H D Tomahawk Kaphaem Road LLC corporate headquarters\ndecompose:\nFACT1: ENTITY1 is H D Tomahawk Kaphaem Road LLC.\nFACT2: ENTITY1 is a child organization.\nFACT3: ENTITY2 is the corporate headquarters of ENTITY1.\nQuestion: Where is ENTITY2?\n\nsentence:\n%s\ndecompose:" % (sentence),
        temperature=0,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["sentence:"]
    )
    response = response['choices'][0]['text']
    sub_sentences = [sub_sentence.strip() for sub_sentence in response.split('\n') if sub_sentence.strip() != '']
    if not all([sub_sentence.startswith('FACT') or sub_sentence.startswith('Question') for sub_sentence in sub_sentences]):
        return None
    return sub_sentences


data_file = './data/ComplexWebQuestions/ComplexWebQuestions_train.json.pkl.T5-paraphrase'
save_file = '%s.codex-decompose' % (data_file)
try:
    data = pickle.load(open(save_file, 'rb'))
except:
    data = pickle.load(open(data_file, 'rb'))
    random.shuffle(data)

num_load_examples, num_save_examples = 0, 0
for item in tqdm(data):
    num_load_examples += 1

    sentence = item['synthesis_question_paraphrases'][0]
    if 'synthesis_question_paraphrases_sub_sentences' not in item:
        try:
            sub_sentences = query_gpt3(sentence)
        except:
            time.sleep(60)
            sub_sentences = query_gpt3(sentence)
        if sub_sentences is None:
            sub_sentences = []
        item['synthesis_question_paraphrases_sub_sentences'] = []
        item['synthesis_question_paraphrases_sub_sentences'].append({'paraphrase': sentence, 'sub_sentences': sub_sentences})

    if item['synthesis_question_paraphrases_sub_sentences'][0]['sub_sentences'] != []:
        num_save_examples += 1

    print('=' * 20)
    print('num_load_examples:', num_load_examples)
    print('num_save_examples:', num_save_examples)
    print('-' * 20)
    print('sentence:', sentence)
    print('')
    for sub_sentence in item['synthesis_question_paraphrases_sub_sentences'][0]['sub_sentences']:
        print(sub_sentence)

    if num_save_examples % 10 == 0:
        pickle.dump(data, open(save_file, 'wb'))

pickle.dump(data, open(save_file, 'wb'))
print('load %d examples' % (num_load_examples))
print('save %d examples' % (num_save_examples))

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
        prompt="sentence:\nWhich has the smallest number of population, the Dunfermline or Luton, either?\ndecompose:\nFACT1: ENTITY1 is the Dunfermline.\nFACT2: ENTITY2 is Luton.\nQuestion: Which has the smallest number of population, ENTITY1 or ENTITY2, either?\n\nsentence:\nWhich one is shorter, The Cassandra Crossing or Road to Perdition?\ndecompose:\nFACT1: ENTITY1 is The Cassandra Crossing.\nFACT2: ENTITY2 is Road to Perdition.\nQuestion: Which one is shorter, ENTITY1 or ENTITY2?\n\nsentence:\nThe publication date of the visual work named The Brothers Grimm is 2005-10-06 where it is published\ndecompose:\nFACT1: ENTITY1 is a visual work.\nFACT2: ENTITY1 is named The Brothers Grimm.\nFACT3: ENTITY1's publication date is 2005-10-06.\nQuestion: Where is ENTITY1 published?\n\nsentence:\nMy Week with Marilyn uses a Visa number 133047, and what's the beginning time?\ndecompose:\nFACT1: ENTITY1 is My Week with Marilyn.\nFACT2: ENTITY1 uses a Visa number 133047.\nQuestion: What's the beginning time of FACT2?\n\nsentence:\nJohn Adams (who's got Laura Linney as part of the cast) has Tom Hollander as part of the cast, what's his role as part of this work?\ndecompose:\nFACT1: ENTITY1 is John Adams.\nFACT2: ENTITY2 is Laura Linney.\nFACT3: ENTITY2 is part of the cast of ENTITY1.\nFACT4: ENTITY3 is Tom Hollander.\nFACT5: ENTITY3 is part of the cast of ENTITY1.\nQuestion: What's ENTITY3's role as part of ENTITY1?\n\nsentence:\nThe university in which Matt Lucas receives his education isn't the time of its origin in 780?\ndecompose:\nFACT1: ENTITY1 is a university.\nFACT2: ENTITY2 is Matt Lucas.\nFACT3: ENTITY2 receives his education in ENTITY1.\nFACT4: The time of ENTITY1's origin isn't 780.\nQuestion: Is FACT4 true?\n\nsentence:\nHow Texas (whose original language is English) has a relationship with Glenn Ford Glenn Ford\ndecompose:\nFACT1: ENTITY1 is Texas.\nFACT2: ENTITY1's original language is English.\nFACT3: ENTITY2 is Glenn Ford Glenn Ford.\nQuestion: How ENTITY1 has a relationship with ENTITY2?\n\nsentence:\nIs Martine McCutcheon's native tongue name the same as Jeremy Renner?\ndecompose:\nFACT1: ENTITY1 is Martine McCutcheon.\nFACT2: ENTITY1's native tongue name is ENTITY2.\nFACT3: ENTITY3 is Jeremy Renner.\nQuestion: Is ENTITY2 the same as ENTITY3?\n\nsentence:\nThe population of Warren County (part of which shares a border with Cattaraugus County) is 40885 at what time it happens\ndecompose:\nFACT1: ENTITY1 is Warren County.\nFACT2: ENTITY2 is Cattaraugus County.\nFACT3: Part of ENTITY1 shares a border with ENTITY2.\nFACT4: ENTITY1's population is 40885.\nQuestion: At what time FACT4 happens?\n\nsentence:\nHow many medals did James Stewart (who won the 13th Academy Award) receive, whose country is the United States of America\ndecompose:\nFACT1: ENTITY1 is James Stewart.\nFACT2: ENTITY1 won the 13th Academy Award.\nFACT3: ENTITY2 is a medal.\nFACT4: ENTITY2's country is the United States of America.\nFACT5: ENTITY1 received ENTITY2.\nQuestion: How many ENTITY2?\n\nsentence:\nWho has the smallest area between independently run cities and is part of the Washington metropolitan area?\ndecompose:\nFACT1: ENTITY1 is an independently run city.\nFACT2: ENTITY1 is part of the Washington metropolitan area.\nQuestion: Which ENTITY1 has the smallest area?\n\nsentence:\nHow many organisations that are big expectations production company (which has Robbie Coltrane as a member of the team) or are owned by Metro-Goldwyn-Mayer\ndecompose:\nFACT1: ENTITY1 is an organisation.\nFACT2: ENTITY2 is big expectations production company.\nFACT3: ENTITY1 is ENTITY2.\nFACT4: ENTITY3 is Robbie Coltrane.\nFACT5: ENTITY3 is a member of the team of ENTITY2.\nFACT6: ENTITY1 is owned by Metro-Goldwyn-Mayer.\nFACT7: FACT3 or FACT6.\nQuestion: How many ENTITY1?\n\nsentence:\nHow many human settlement that contains Montclair State University or that is the death place of Donald M. Payne\ndecompose:\nFACT1: ENTITY1 is a human settlement.\nFACT2: ENTITY1 contains Montclair State University.\nFACT3: ENTITY1 is the death place of Donald M. Payne.\nFACT4: FACT2 or FACT3.\nQuestion: How many ENTITY1?\n\nsentence:\nFor a tabletop play named after War on Terror (which is the main subject of the book Homeland), is it due out in 2006?\ndecompose:\nFACT1: ENTITY1 is a tabletop play.\nFACT2: ENTITY2 is War on Terror.\nFACT3: ENTITY1 is named after ENTITY2.\nFACT4: ENTITY2 is the main subject of the book Homeland.\nQuestion: Is ENTITY1 due out in 2006?\n\nsentence:\nHow many county of Oregon that is located in or next to Antarctica (the one that is the main subject of March of the Penguins)\ndecompose:\nFACT1: ENTITY1 is a county of Oregon.\nFACT2: ENTITY2 is Antarctica.\nFACT3: ENTITY1 is located in or next to ENTITY2.\nFACT4: ENTITY2 is the main subject of March of the Penguins.\nQuestion: How many ENTITY1?\n\nsentence:\n%s\ndecompose:" % (sentence),
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


data_file = './data/KQA/train.json.pkl.T5-paraphrase'
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

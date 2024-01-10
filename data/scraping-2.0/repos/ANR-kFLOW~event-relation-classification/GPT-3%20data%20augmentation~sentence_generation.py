import random
import time

import openai
import pandas as pd


def read_file(path):
    file = pd.read_csv(path)
    return file


enable = read_file(
    '/Users/youssrarebboud/Documents/GitHub/EventRelationDataset/annotation_csv/well aligned rows Timebank/intends.csv')

rslt_df = enable[enable['annotation'] == 1]
enable_examples = set(rslt_df.sentence.values)
generated_Sentences = []
request_enable = 'a condition is The fact of having certain qualities, which may trigger events,  an event is A possible or actual event, which can possibly be defined by precise time and space cordinates ""  enables relationship Connects a condition or an event (trigger1),  with an  other event (trigger 2),it is contributing to realize as an enabling factor."" give me very long political example sentences follwong these examples and give me each sentence in one line please'
request_prevents = 'an event is A possible or actual event, which can possibly be defined by precise time and space cordinates, prevention is a relation between an event and the event for which is the cause of not happening.Example: the strike was sufficient to block the changement in working conditions. give me very long political different in topics example sentences which have the prevention relationship and give me each sentence in one line please, for example'
request_intention = 'an event is A possible or actual event, which can possibly be defined by precise time and space cordinates, Connects an Event with the effect it is intended to cause (independently if the result is achieved or not).Example: The government voted a law, in the attempt of reducing unemployment. give me very long political different in topics example sentences which have the intention relationship and give me each sentence in one line please, for example'
while len(generated_Sentences) < 80:
    prompt = request_intention + ','.join(random.sample(list(enable_examples) + generated_Sentences, 5))
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=2000,
        # top_p=1,
        # frequency_penalty=0.0,
        # presence_penalty=0.0,
        # stop=["\n"]
    )
    print('here')
    # print(response['choices'][0]['text'])
    time.sleep(3)
    for sen in response['choices'][0]['text'].split('\n'):

        if len(sen) > 0:
            generated_Sentences.append(sen)
            print(sen)

    generated_Sentences_df = pd.DataFrame(list(set(generated_Sentences)))
    generated_Sentences_df.to_csv('/Users/youssrarebboud/Desktop/intention_left.csv')
    # for x in response['choices'][0]['text'].split('\n'):
    #     print(x)
    #     generated_sentences.append(x)

#     generated_sentences.append(response['choices'][0]['text'])
#   response = response = bot.ask(request_enable+','.join(random.sample(list(enable_examples)+generated_Sentences,5)))
#   for sen in response.split('\n'):
#
#       print(sen)
#
#       generated_Sentences.append(sen)
#

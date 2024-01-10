import time

import openai
import pandas as pd


# openai.api_key = os.getenv("OPENAI_API_KEY")
#
# openai.Engine.list()


# write your file name instead of jokes_prepared.jsonl
# with open("/Users/youssrarebboud/Downloads/prompt_prepared.jsonl") as f:
#       response = openai.File.create(file=f, purpose='fine-tune')
# print(response)
def read_file(path):
    file = pd.read_csv(path)
    return file


def random_selection(file):
    df_elements = file.sample(n=7)
    return df_elements['sentence']


generated_sentences = []

file = read_file('/Users/youssrarebboud/Desktop/intention_left.csv')
print(len(file))
file.dropna(axis=0, how='any', subset=None, inplace=True)
file = file.drop_duplicates()
print(len(file))
file.columns = ['idx', 'sentence']
prompt_intention = "an event is A possible or actual event, which can possibly be defined by precise time and space coordinates ""  intention relationship Connects an event (trigger1),  with an  other event (trigger 2), that is intended to cause it  independetly if the result is achieved or not ""so if:The government voted a law, in the attempt of reducing unemployment.''  is an sentence that has an intention relationship between the event(voted)==(trigger1) and the event (reducing)==trigger2"" and also this sentence The company said it expects to use the proceeds to repay certain bank debt and for general corporate purposes, including establishing new operating centers and possible acquisitions, with trigger1==use and trigger2== establishing,what would be the trigger1 and trigger2 in  these sentences, give me only one single word for each trigger an only two triggers per sentence, put each pair between parentheses in a separate line:"
prompt_prevention = 'an event is A possible or actual event, which can possibly be defined by precise time and space cordinates, a prevention relationship Connect an event (trigger1) with the event (trigger 2) for which is the cause of not happening. so if in this sentence Subcontractors  will  be offered a settlement and a swift transition to new management  is expected  to avert an exodus of skilled workers from Waertsilae Marine\'s two big shipyards, government officials said. is an expression with prevention relationship between  settelement(trigger1) and oxodus(trigger2), what would be the trigger1 and trigger2 in  these sentences, give me only one single word for each trigger an only two triggers per sentence, put each pair between parentheses in a separate line: '
prompt_enable = "a condition is The fact of having certain qualities, which may trigger events,  an event is A possible or actual event, which can possibly be defined by precise time and space cordinates ""  enables relationship Connects a condition or an event (trigger1),  with an  other event (trigger 2),it is contributing to realize as an enabling factor.""so if:the basketball player is so tall that he was scoring many times during the match''  is an sentence that has an enabling relationship between the Condition(tall)==(trigger1) and the event (scoring)==trigger2"" and also this sentence In addition, Courtaulds said the moves are logical because they will allow both the chemicals and textile businesses to focus more closely on core activities. with trigger1==moves and trigger2== focus,what would be the trigger1 and trigger2 in  these sentences, give me only one single word for each trigger an only two triggers per sentence, put each pair between parentheses in a separate line:"
print(file)

#
# # Here set parameters as you like

for i, row in file.groupby(file.index // 1):
    examples = row['sentence']
    my_prompt = prompt_intention + ' '.join(examples)

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=my_prompt,
        temperature=0,
        max_tokens=2000,
        # top_p=1,
        # frequency_penalty=0.0,
        # presence_penalty=0.0,
        # stop=["\n"]
    )

    # print(response['choices'][0]['text'])
    # print(response['choices'][0]['text'])
    # for x in response['choices'][0]['text'].split('\n'):
    #     print(x)
    #     generated_sentences.append(x)

    generated_sentences.append(response['choices'][0]['text'])
    time.sleep(10)
    data_frame = pd.DataFrame(generated_sentences, columns=['generated events'])
    print('I am just here ')
    data_frame.to_csv('left_intention_with_events2.csv')
    print('saved')

# data_frame = pd.DataFrame(generated_sentences, columns=['generated sentences'])
# data_frame.to_csv('generated_event_triggers_GPT3_second_hit_intention.csv')

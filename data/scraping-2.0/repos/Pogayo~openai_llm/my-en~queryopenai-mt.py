
# strategy
import os
import openai
from time import sleep

openai.api_key = "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"

names_list = [name.strip() for name in open("names.tsv").readlines()]
names = [" ".join(name.split('\t')) for name in names_list]

out_file = open("parallel-my_en", "a")

prompt = 'I will give you a name in Burmese and its English translation.\
 Construct sentences using them in Burmese, completely in Burmese script and provide their translations in English.  For example,\
For names:စီမင်နော့ဖ Siminoff, အဂိုစလင်း Gosling return: \
Burmese: ၂၀၁၇ ခုနှစ်နှောင်းပိုင်းတွင် စီမင်နော့ဖ်က စျေးဝယ် ရုပ်မြင်သံကြားလိုင်း  တွင် ပေါ်လာပါသည်။ \n \
English: In late 2017, Siminoff appeared on shopping television channel .\n \
Burmese: ဂိုစလင်းနှင့် စတုန်းတို့သည် အကောင်းဆုံး အမျိုးသားနှင့် အမျိုးသမီး ဇာတ်ဆောင်ဆုများအတွက် ဆန်ခါတင် စာရင်းတင်သွင်းခြင်းကို အသီးသီးရရှိခဲ့ကြသည်။ \n \
English: Gosling and Stone received nominations for Best Actor and Actress, respectively. \n \
 Try to use sentence structures and meanings to showcase your understanding of language.\
  For example, a sentence could be a simple statement, a complex compound sentence,\
   a question, or an exclamation. Remember to have the Burmese sentence completely in Burmese script. Only return one example and use Burmese:  and English: as language  prefix'
# prompt1 = instruct + "Sentence 1: " + examples[0] + "\n"
i = 35100
while i < 35500:
    names_current = names[i]+ "\n"+names[i+1] +"\n"+names[i+2]
    instruct = prompt + "\n Your turn: " + names_current
    # print(instruct)

    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                              messages=[{"role": "user", "content": instruct}])
    result = completion.choices[0].message.content
    # print(result.strip()+"\n")

    # response = openai.chat.Completion.create(
    #     model="gpt-3.5-turbo-0301",
    #     # model="text-curie-001",
    #     prompt=instruct,
    #     temperature=0.1,
    #     max_tokens=300,
    #     top_p=1,
    #     frequency_penalty=0,
    #     presence_penalty=0
    # )
    # result = response['choices'][0]['text']
    # print(result)
    out_file.write(result.strip()+"\n")
    sleep(2)
    i += 4

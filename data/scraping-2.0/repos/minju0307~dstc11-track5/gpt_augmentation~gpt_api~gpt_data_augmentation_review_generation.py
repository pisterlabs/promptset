import openai
import re
import os

## api key
OPENAI_API_KEY="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"
openai.api_key=OPENAI_API_KEY

## read file
train_history=open('train_dialgoue_history_for_dataaug.txt').readlines()
train_last_question=open("train_last_ques_aug_final.txt").readlines()
val_history=open('val_dialgoue_history_for_dataaug.txt').readlines()
val_history=val_history*100
val_response=open('val_last_question_for_dataaug.txt').readlines()
val_response=val_response*100
val_knowledge=open('knowledge_val.txt').readlines()
val_knowledge=val_knowledge*100

val_example=[]
train_example=[]

for idx, p in enumerate(val_history):
    example=''
    if p.strip():
        example += '##\n'
        example += p +'\n'
        example += "based on dialogue history and user's last question, generate rivews or FAQs to answer user's last question" + '\n'
        example += "user's last question: " + val_response[idx] + '\n'
        example += "reviews or FAQs: " + val_knowledge[idx] + '\n\n'
        val_example.append(example)

for idx, p in enumerate(train_history):
    example=''
    if p.strip():
        example += '##\n'
        example += p +'\n'
        example += "based on dialogue history and user's last question, generate rivews or FAQs to answer user's last question" + '\n'
        example += "user's last question: " + train_last_question[idx] + '\n'
        example += "reviews or FAQs: "
        train_example.append(example)

print(len(train_example))

## api call

model = "gpt-3.5-turbo"

ini=0

while True:
    iidx = []

    for idx,text in enumerate(train_example[ini:], start=ini):

        ## example
        query = ''
        for i in range(4):
            query += val_example[idx+i]

        ## question
        query += text
        
        messages = [
            {"role": "system", "content": "you are a professional dialogue data augmentator."},
            {"role": "user", "content": query}
        ]

        try:
            response = openai.ChatCompletion.create(model=model, messages=messages)
            answer = response['choices'][0]['message']['content']
            print(idx)
            print(query)
            print()

            output_dir = './review_generation_train'
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            with open(f'{output_dir}/review_train_{idx}.txt', 'w') as fw:
                fw.write(answer)

            iidx.append(idx)

        except:
            print(f"{idx}_error")
            break

    ini=ini+len(list(set(iidx)))
    if ini>=len(train_history):
        break

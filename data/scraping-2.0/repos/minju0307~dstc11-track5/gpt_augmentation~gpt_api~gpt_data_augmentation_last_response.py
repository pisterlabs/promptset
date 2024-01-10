import openai
import re
import os

## api key
OPENAI_API_KEY="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"
openai.api_key=OPENAI_API_KEY

## train files
train_history=open('train_dialgoue_history_for_dataaug.txt').readlines() ## 여태껏 나온 것들을 먼저 하기
train_last_question=open("train_last_ques_aug_final.txt").readlines()
train_review=open("train_reivews_aug_ver2.txt").readlines()

## val files
val_history=open('val_dialgoue_history_for_dataaug.txt').readlines()
val_history=val_history*100
val_last_question=open('val_last_question_for_dataaug.txt').readlines()
val_last_question=val_last_question*100
val_gold_response=open('val_gold_responses.txt').readlines()
val_gold_response=val_gold_response*100
val_knowledge=open('knowledge_val.txt').readlines()
val_knowledge=val_knowledge*100

val_example=[]
train_example=[]

for idx, p in enumerate(val_history):
    example=''
    if p.strip():
        example += '##\n'
        example += "reviews or FAQs: " + val_knowledge[idx] + '\n'
        example += p +'\n'
        example += "based on reviews or FAQs, answer to user's last question. you should write in 32 tokens." + '\n'
        example += "user's last question: " + val_last_question[idx] + '\n'
        example += "S: "+val_gold_response[idx] +'\n'
        val_example.append(example)

for idx, p in enumerate(train_history):
    example=''
    if p.strip():
        example += '##\n'
        example += "reviews or FAQs: "+train_review[idx]+'\n'
        example += p +'\n'
        example += "based on reviews or FAQs, answer to user's last question. you should write in 32 tokens." + '\n'
        example += "user's last question: " + train_last_question[idx] + '\n'
        example += "S: "
        train_example.append(example)

print(len(train_example))

## api call

model = "gpt-3.5-turbo"

ini=11359

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
            {"role": "system", "content": "you are a knowledge-based dialogue chatbot system."},
            {"role": "user", "content": query}
        ]

        try:
            response = openai.ChatCompletion.create(model=model, messages=messages)
            answer = response['choices'][0]['message']['content']
            print(idx)
            print()

            output_dir = './last_system_response_train'
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            with open(f'{output_dir}/response_train_{idx}.txt', 'w') as fw:
                fw.write(answer)

            iidx.append(idx)

        except:
            print(f"{idx}_error")
            break

    ini=ini+len(list(set(iidx)))
    if ini>=len(train_history):
        break

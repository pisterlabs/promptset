import openai
import re
import os

## api key
OPENAI_API_KEY="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"
openai.api_key=OPENAI_API_KEY

## read file
train_history=open('train_dialgoue_history_for_dataaug.txt').readlines()
val_history=open('val_dialgoue_history_for_dataaug.txt').readlines()
val_history=val_history*100
val_response=open('val_last_question_for_dataaug.txt').readlines()
val_response=val_response*10000

train_h=[]
val_h=[]
val_r=[]

for idx, p in enumerate(train_history):
    if p.strip():
        train_h.append(p)

for idx, p in enumerate(val_history):
    if p.strip():
        val_h.append(p)
        val_r.append(val_response[idx])

print(len(train_h))
print(len(val_h))
print(len(val_r))

## api call

model = "gpt-3.5-turbo"

ini=2125

while True:
    iidx = []

    for idx,text in enumerate(train_h[ini:], start=ini):

        ## example
        query = ''
        for i in range(4):
            query += "##"
            query += val_h[idx+i]
            query += "based on dialogue context, generate last question. it should be need knowledge such as reviews or FAQs."
            query += val_r[idx+i]

        ## question
        query += "##"
        query += train_h[idx]
        query += "based on dialogue context, generate last question. it should be need knowledge such as reviews or FAQs."
        query += "U: "
        
        messages = [
            {"role": "system", "content": "you are a professional dialogue data augmentator."},
            {"role": "user", "content": query}
        ]

        try:
            response = openai.ChatCompletion.create(model=model, messages=messages)
            answer = response['choices'][0]['message']['content']
            print(idx)
            print()

            output_dir = './last_question_generation_train'
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            with open(f'{output_dir}/last_question_train_{idx}.txt', 'w') as fw:
                fw.write(answer)

            iidx.append(idx)

        except:
            print(f"{idx}_error")
            break

    ini=ini+len(list(set(iidx)))
    if ini>=len(train_h):
        break

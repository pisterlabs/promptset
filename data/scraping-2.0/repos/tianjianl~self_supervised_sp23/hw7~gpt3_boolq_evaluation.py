import os
import openai
from datasets import load_dataset

openai.api_key = os.getenv("OPENAI_API_KEY")
boolq = load_dataset("boolq")

context = ""
for question, answer, passage in zip(boolq["train"][8:15]['question'], boolq["train"][8:15]['answer'], boolq["train"][8:15]['passage']):
    context += question + "?" + "\n" + passage + "\n" + str(answer) + "\n\n" 

print(context)

#validate on 30 instances of BoolQ validation set
correct_count = 0
for question, answer, passage in zip(boolq["validation"][0:30]['question'], boolq["validation"][0:30]['answer'], boolq["validation"][0:30]['passage']):
    prompt = question + "?" + "\n" + passage + "\n" 
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=context + prompt,
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    print(response['choices'][0]['text'])
    print(answer)
    if response['choices'][0]['text'] == str(answer):
        correct_count += 1
    print("-----------------------------")
print(f"accuracy = {correct_count/30}")

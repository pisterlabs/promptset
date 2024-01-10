import openai
import os
import json

# create a file named apikey.txt and paste your api key in it
openai.api_key_path="./apikey.txt"

def generatePrompt(ans1, ans2):
    return "Below is a sentence and array of strings. Check if the sentence means the same as those other strings. The sentence might be a much general statement of those strings.\nsentence :" + ans1 + "strings :" + ans2 + "\nAnswer with rating bewteen 0 to 10, it must be integers only: "

def generate(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=100,
    )
    return response

# read the json file 
with open('answers.json') as f:
    data = json.load(f)

# get the answers from the json file
ans1 = data['studentAns']
ans2 = data['chatGPTAns']

# compare ans1 with every string in ans2
scores=[]
for i in ans2:
    prompt = generatePrompt(ans1, i)
    response = generate(prompt)
    rating=response['choices'][0]['text']
    scores.append(int(rating))

# return the maximum score
print(max(scores))

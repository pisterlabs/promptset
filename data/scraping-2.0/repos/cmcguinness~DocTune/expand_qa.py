"""
    When fine-tuning, it's helpful to have variations of the user's questions to make sure
    we have enough examples of how people might ask the same question.

    This program reads in our Q&A document and asks GPT to generate 10 variations for us.
    We then write out the original question and answer as well as 10 new Q&As (where the
    answer remains the same).
"""
import os
import json
import openai
from readdocs import readdocs

print("Directories in inputs:")
dirs = os.listdir('inputs')
for d in dirs:
    print(d)

project = input('Please enter the subdirectory that holds your inputs: ')
# The source
qa_list = readdocs(f'inputs/{project}/OriginalQandA.txt')

# Make sure you have the environment variable set before running!
openai.api_key = os.getenv("OPENAI_API_KEY")

# Where we write the expanded Q&A to
new_docs = open(f'inputs/{project}/QandA.txt','w')

# How we instruct GPT to generate our variations
sys_prompt = 'You are a system that creates variations of text while retaining the meaning.  I will enter some text and you are to give me back 10 variations of the text in a JSON array with the following structure:\n\n{\n  "variations" : []\n}'


# Helper function to write out a new Q&A
def write_qa(q:str, a:str):
    global new_docs
    new_docs.write('# Q\n')
    new_docs.write(q+'\n')
    new_docs.write('# A\n')
    new_docs.write(a+'\n')


# Helper function to call GPT and extract the variatiosn
def ask_gpt(system: str, user: str):
    p_model = "gpt-3.5-turbo"
    p_temperature = 0
    completion = openai.ChatCompletion.create(
        model=p_model,
        temperature=p_temperature,
        messages=[
            {
                "role": "system",
                "content": system
            },
            {
                "role": "user",
                "content": user
            }
        ]
    )
    resp = completion.choices[0].message["content"]
    return json.loads(resp)['variations']


# Loop over each Q&A in our list and perform the magic!
for qa in qa_list:
    write_qa(qa['q'].strip(), qa['a'].strip())
    print(f'Q: {qa["q"]:32}...')
    variations = ask_gpt(sys_prompt, qa['q'])
    for v in variations:
        print(f'     V: {v:32}...')
        write_qa(v, qa['a'].strip())

# Done!
new_docs.close()



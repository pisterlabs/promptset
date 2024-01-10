import pandas as pd
import sys,os,subprocess,argparse
from openai import OpenAI
import time

argparser = argparse.ArgumentParser(description='Get AI descriptions from protein descriptions')

argparser.add_argument('-i', '--input', help='Input csv file (Must have a column called Description with protein descriptions)', required=True)
argparser.add_argument('-o', '--output', help='Output csv file', required=True)

args = argparser.parse_args()

test_file = pd.read_csv(args.input)
output_file = args.output

import time
test_file
client=OpenAI()
assistant= client.beta.assistants.create(
    name="Protein Classifier",
    instructions="Based on the following description of a protein, I want you to give a short description of the protein's function with a 4 word limit",
    model='gpt-3.5-turbo'
)
#iterate through descriptions and get responses in matching column
test_file['Function']=''
for i in range(len(test_file)):
    thread=client.beta.threads.create()
    description=test_file['Description'][i]
    message=client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=description
    )
    run= client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )
    time.sleep(5)
    messages = client.beta.threads.messages.list(
    thread_id=thread.id
    )
    assistant_response=''
    for message in messages.data:
        if message.role == "assistant":
            assistant_response = message.content[0].text.value
            break
    print(assistant_response)
    test_file['Function'][i]=assistant_response
    time.sleep(5)
    #clear thread
    client.beta.threads.delete(thread.id)
    #clear thread

test_file.to_csv(output_file,index=False)

# request_utils.py
# contains the methods to make sending requests to the API easier

import openai
import os

TEXT_COMPLETE_MODEL="text-davinci-003"
#CODE_COMPLETE_MODEL="code-davinci-002"
#EDIT_MODEL="text-davinci-edit-001"
openai.api_key = os.getenv("OPENAI_API_KEY")

def apireq_text_completion(req_prompt, req_max_tokens, vrbs):
    if vrbs == 1:
        print("Prompt: " + req_prompt + "\n")
        print("Requesting from API...\n")

    response = openai.Completion.create(
        model=TEXT_COMPLETE_MODEL,
        prompt=req_prompt,
        temperature=0,
        max_tokens=req_max_tokens,
        top_p=1,
        frequency_penalty=0.5,
        presence_penalty=0)

    if vrbs == 1:
        print("Text recieved: " + response['choices'][0]['text'])

    #print("")
    return response


import openai
# import config
#import api_interact.PromptHandler as PromptHandler

openai.api_key= "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"

def gen_from_prompt(privprompt, temperature,):
    prompt = privprompt    
    response = openai.Completion.create(engine="text-davinci-001", prompt=prompt, max_tokens=100)
    return response.choices[0]['text']

print(gen_from_prompt("hello ai"))

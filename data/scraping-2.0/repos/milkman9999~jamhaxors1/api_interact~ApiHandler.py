import openai
from api_interact import config

# set key from config
openai.api_key=config.API_KEY

def gen_from_prompt(privprompt, temperature, max_tokens):
    prompt = privprompt    
    response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=max_tokens)
    return response.choices[0]['text']

def response_check(responsetocheck, temperature, max_tokens):
    checkmsg = "Return 'true' if the following text contains a list. Else, return 'false': {}".format(responsetocheck)
    response = openai.Completion.create(engine="text-davinci-003", prompt=checkmsg, max_tokens=max_tokens)
    if(response.choices[0]['text']).__contains__('True'):
        return True
    else:
        return False
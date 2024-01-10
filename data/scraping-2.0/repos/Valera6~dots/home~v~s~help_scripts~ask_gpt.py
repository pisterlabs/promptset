# TODO: start keeping a json with current conversation in /tmp. Always write to it; but only load when `qc` command is callled instead of `q`

import openai
import requests
import json
import time
import sys
import os

api_key = os.getenv("OPENAI_KEY")
# second var is cost of 1k tokens
gpt35 = ('gpt-3.5-turbo', 0.002)
gpt4t = ('gpt-4-1106-preview', 0.015) # price is a guess, and is not to be trusted

standard_instruction = """Respond concisely. If no context provided, question is about Linux (arch) cli tools. No talk; just go"""
f_instruction = """You are my programmer buddy. I'm pissed beyond words at some tool. You are to fully agree with me and encourage my righteous anger, using one sentence, very profane language, and insulting design flaws specific to the shit code we're angry at"""

# ==========================================================
openai.api_type = 'azure'
openai.api_key = api_key


def request(question, model=gpt4t, debug=False):
    global standard_instruction, f_instruction
    opt = sys.argv[1]
    if opt == '-s':
        instruction = standard_instruction
    elif opt == '-f':
        instruction = f_instruction
        question = "fuck " + question
    else:
        print('no such option, think again')
        sys.exit(1)
    system_line = {"role": "system", "content": instruction}
    user_line = {"role": "user", "content": question}
    conversation = [system_line] + [user_line]

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    data = {
        "model": f"{model[0]}",
        "messages": conversation,
        "temperature": 0,
        #"max_tokens": 100,
    }
    start_time = time.time()
    r = requests.post(url, headers=headers, data=json.dumps(data)).json()
    end_time = time.time()

    if debug:
        print(f'{model[0]}:')
        print(f"{conversation[-1]['content']}")
        print(f"{r['choices'][0]['message']['content']}")
        tokens = float(r['usage']['total_tokens'])
        cost = model[1] * tokens/1000
        print(f"cost: {cost} in {end_time-start_time}\n")
    return r['choices'][0]['message']['content'].split(';')[0]


def main():
    question = ' '.join(sys.argv[2:])
    return request(question)


if __name__ == '__main__':
    print(main())

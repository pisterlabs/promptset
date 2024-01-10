import flask
from flask import request
import logging
import json
import openai
import emoji
from time import time,sleep
from random import seed, uniform


log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
app = flask.Flask('gpt3')
with open('openaiapikey.txt', 'r') as infile:
    open_ai_api_key = infile.read()
openai.api_key = open_ai_api_key


def gpt3_completion(prompt, prompt_name, engine='curie', temp=0.7, top_p=0.5, tokens=100, freq_pen=0.5, pres_pen=0.5, stop=['<<END>>', '\n\n']):
    max_retry = 5
    retry = 0
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=emoji.demojize(prompt),
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)
            text = emoji.demojize(response['choices'][0]['text'].strip())
            filename = '%s_%s.txt' % (time(), prompt_name)
            with open('gpt3_logs/%s' % filename, 'w') as outfile:
                outfile.write('PROMPT:\n' + prompt + '\n\nRESULT:\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return None
            print('Error communicating with OpenAI:', oops)
            sleep(uniform(0.05, 0.25))  # sleep a fraction of a second and then try again


@app.route('/', methods=['POST'])
def completion():
    payload = request.json
    print('\n\nPayload:', payload)
    text = gpt3_completion(payload['prompt'], payload['prompt_name'])
    print('\n\nResponse:', text)
    return text


if __name__ == '__main__':
    print('Starting Transformer Svc')
    seed()
    app.run(host='0.0.0.0', port=7777)
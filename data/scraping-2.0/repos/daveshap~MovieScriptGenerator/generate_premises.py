import openai
from time import sleep
import re


with open('openaiapikey.txt', 'r') as infile:
    openai.api_key = infile.read()


def gpt3_completion(prompt, engine='text-davinci-002', temp=1.1, top_p=1.0, tokens=500, freq_pen=0.0, pres_pen=0.0, stop=['asdfasdf']):
    max_retry = 5
    retry = 0
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,         # use this for standard models
                #model=engine,           # use this for finetuned model
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)
            text = response['choices'][0]['text'].strip()
            text = re.sub('\s+', ' ', text)
            #save_gpt3_log(prompt, text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return None
            print('Error communicating with OpenAI:', oops)
            sleep(1)


if __name__ == '__main__':
    for i in range(176, 201):
        with open('prompt_premise.txt', 'r') as infile:
            prompt = infile.read()
        premise = gpt3_completion(prompt)
        print('\n\n\n', premise)
        filename = "premise_mystery_%s.txt" % i
        with open('premises/%s' % filename, 'w', encoding='utf-8') as outfile:
            outfile.write(premise)
        #exit(0)
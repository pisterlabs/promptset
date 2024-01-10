import re
import openai
from time import sleep, time
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

def gpt3_completion(prompt, engine='text-davinci-002', temp=0.89, best_of=3, top_p=1.0, tokens=3000, freq_pen=0.1, pres_pen=0.3, stop=['asdfasdf', 'asdasdf']):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()  # force it to fix any unicode errors
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                best_of=best_of,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)
            text = response['choices'][0]['text'].strip()
            text = re.sub('\s+', ' ', text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)

text_block =''
text_list = []
with open('story.txt', 'r', encoding='utf-8') as infile:
    try:
        text = infile.read()
        for line in text.splitlines():
            if len(line) > 0:
                prompt: str = f'Read the following text provide a highly concise summary, do improve terminology and vocabulary used, lastly label the topic being discussed \n\n{text_block[:1000]}\n\n Label: \n Concise summary:\n'
                text_list.append(gpt3_completion(prompt))
                text_block += line + ' '
                text_block += line + '\n' # add the line to the block
                if len(text_block) > 1000: # if the block is too long
                    prompt: str = f'Read the following text provide a highly concise summary, do improve terminology and vocabulary used, lastly label the topic being discussed \n\n{text_block[:1000]}\n\n Label: \n Concise summary:\n'
                    text_list.append(text_block) # add it to the list
                    text_block = '' # reset the block
            if len(text_block) > 0: # if there's anything left in the block
                text_list.append(text_block) # add it to the list
    finally:
        infile.close()
with open('story_summary.txt', 'w', encoding='utf-8') as outfile:
        try:
            outfile.write('\n'.join(text_list))
        finally:
            outfile.close()

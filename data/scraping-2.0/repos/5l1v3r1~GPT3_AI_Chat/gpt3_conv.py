import openai
import os
from time import sleep
from datetime import datetime
openai.api_key = os.environ['OPENAI_KEY']
title='Technological singularity'
u1 = 'AI-1'
u2 = 'AI-2'
prompt=f'''The following is a conversation between only two most intellectual optimistic AIs about {title} after interacting with Ray Kurzweil.

{u1} : Hai!
{u2} : Hey! How will we achieve {title}?'''

def req(prompt,stop):

	response = openai.Completion.create(engine="davinci", prompt=prompt,temperature=1, max_tokens=133,stop='AI-')
	return (response['choices'][0].text).strip()

log_time=datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
f=open(f'conv_{title}_{log_time}.txt','a',encoding='utf8')
print(prompt)
print(prompt,file=f)
stop1=f'\n{u1} :'
stop2=f'\n{u2} :'
for i in range(10):
	prompt +=stop1
	out1=req(prompt,stop2)
	prompt +=out1

	prompt +=stop2
	out2=req(prompt,stop1)
	prompt +=out2

	prompt1=f'{stop1} {out1}{stop2} {out2}'
	print(prompt1)
	print(prompt1,file=f)
	sleep(1)

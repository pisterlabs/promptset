import openai
import random
import time
from sqs_wrapper import SQSWrapper
from env_config import (
    QUEUE_NAME,
    OPENAI_API_KEY
) 


topics = [
    "Can you help me to write a brand new random quote regarding power in 140 characters?",
    "Explica sin ser redundante lo que es una architectura hexagonal (less than 140 characters)",
    "Explica sin ser redundante lo que son microservicios (less than 140 characters)",
    "Elaborate a creative idea to improve our investments in less than 140 characters and no hashtags",
    "Explica sin ser redundante lo que es el patron saga (less than 140 characters)",
    "Please Elaborate a creative idea to improve our sales in 140 characters",
    "Explica sin ser redundante la diferencia entre orquestacion y coreografia en arquitectura de software (less than 140 characters)",
    "can you help me to write random quote about war in latin that I can tweet in 140 characters?",
    "Explica sin ser redundante que es un service mesh (less than 140 characters)",
    "tweet a quote from the catholic bible, quran or torah (less than 140 characters)",
    "Menciona sin ser redundante que es un sidecar en arquitectura de software y un ejemplo (less than 140 characters)",
]

def get_openai_api_key():
    api_key = OPENAI_API_KEY
    if api_key is None:
        raise Exception('OpenAI API key not found')
    return api_key

def gpt3(prompt):
    
    openai.api_key = get_openai_api_key()

    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        temperature=0.5,
        max_tokens=140,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    content = response.choices[0].text.split('.')
    #print(content)
    return response.choices[0]["text"]


def job(sqs):
    random_topic = random.randint(0, len(topics)-1)
    quote = gpt3(topics[random_topic]);
    quote = quote.replace('"', '')
    quote = quote.replace('/n', '')
    quote = quote.strip()
    print(quote)
    sqs.send(quote)

def sleep_seconds(seconds):
    time.sleep(seconds)

def sleep_minutes(minutes):
    time.sleep(minutes * 60)

def main():
    
    sqs = SQSWrapper(QUEUE_NAME)
    while True:
        random_minute = random.randint(5,10)
        job(sqs)
        print(f"...waiting for: {random_minute}")
        sleep_seconds(random_minute)
        #sleep_seconds(random_minute)

if  __name__ == '__main__':
    main()
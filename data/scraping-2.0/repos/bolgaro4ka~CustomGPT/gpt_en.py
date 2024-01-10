import openai
import requests

apis=open('keys.apikey').read().split(', ')
openai.api_key = str(apis[0])
messages = []


def validetor_en(question):
    question=question.split(' ')
    print(question)
    if ('weather' in question) or ('weathers' in question):
        try:
            params = {'q': 'Novodugino,RU', 'units': 'metric', 'lang': 'ru',
                      'appid': str(apis[1])}
            response = requests.get(f'https://api.openweathermap.org/data/2.5/weather', params=params)
            if not response:
                raise
            w = response.json()
            return (f" Weather: {w['weather'][0]['description']} {round(w['main']['temp'])} C")

        except:
            return ('Error! Cheak API-key')
    elif ('TSWW' in question):
        return 'This phrase is a stub to test the program without using ChatGPT! In the depths of the tundra, otters dig cedar kernels into buckets.'
    elif ('/generate_image' in question):
        del question[0]
        response = openai.Image.create(
            size=question[-1],
            prompt=''.join(question[:-1]),
            n=1
        )
        image_url = response['data'][0]['url']
        print(image_url)
        return f'Image {question[:-1]}, size: {question[-1]} generated! Link: {image_url}'
    elif ([''] == question) or ([' ']==question): return 'Empty question!'
    else: return 'OMG'


def answer(question, temperature):
    valid=validetor_en(question)
    print(valid)
    if valid == "OMG":
        message = question  # вводим сообщение
        if message == "quit": return 7


        messages.append({"role": "user", "content": message})
        chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, temperature=temperature)
        reply = chat.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})
        return reply
    else: return valid

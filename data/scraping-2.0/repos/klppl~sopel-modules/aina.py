import sopel
import openai
import requests
import textwrap
import apikeys

@sopel.module.commands('aina')
def aina(bot, trigger):
    # Set up the OpenAI API client - https://beta.openai.com/account/api-keys
    openai.api_key = apikeys.OPENAI_API_KEY

    model_engine = "text-davinci-003"
    prompt = trigger

    completion = openai.Completion.create(
    engine=model_engine,
    prompt=prompt,
    max_tokens=1024,
    n=1,
    stop=None,
    temperature=0.5,
    )

    response = completion.choices[0].text
    response = textwrap.fill(response, width=100)

    response = response.encode("utf-8")
    response = requests.post("https://dumpinen.com", data=response)
    aina_output = response.text

    bot.say(f'AIna: {aina_output}')

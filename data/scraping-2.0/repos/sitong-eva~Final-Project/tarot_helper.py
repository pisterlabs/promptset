import requests
import json
import random
import os
import openai

openai.api_key = os.environ["OPENAI_KEY"]
# Tarot API
url = "https://tarot-api.onrender.com/api/v1/cards"
response = requests.get(url)

data = json.loads(response.text)
# print(data)

cards = data["cards"]
# print(cards)

# List of Cards
lst = []

for card in cards:
    # print(card)
    lst.append(card["name"])
# print(lst)


def draw_love():
    """
    This function draws tarots cards from Lst, to get a tarot spread to predict love.
    Returns 3 cards, for past, present, and future for love life of user.
    https://www.google.com/url?sa=i&url=https%3A%2F%2Fvekkesind.com%2Ftarot-spreads-for-love%2F&psig=AOvVaw1R0lUvivsd1N6vmbr8Xr8n&ust=1682457158030000&source=images&cd=vfe&ved=0CBAQjRxqFwoTCMDLnLy3w_4CFQAAAAAdAAAAABAd
    """
    love_past = random.choice(lst)
    love_present = random.choice(lst)
    love_future = random.choice(lst)
    love = love_past, love_present, love_future
    return love


def draw_yearly_forecast():
    """
    This function draws tarots cards from Lst, to get a tarot spread to predict the yearly forecast.
    Returns 7 cards, for main theme, departing energies, talents, future opportunities, obstacles, dealing with obstacles, and achievements for the yearly forecase of user
    https://www.google.com/url?sa=i&url=https%3A%2F%2Fmywanderingfool.com%2Ftarot%2Ftarot%2Fhow-to-map-out-the-year-ahead-a-new-years-tarot-spread%2F&psig=AOvVaw1FtbMLSaCV9MYNrsBDsoyx&ust=1682457244279000&source=images&cd=vfe&ved=0CBAQjRxqFwoTCOimmOW3w_4CFQAAAAAdAAAAABAY
    """
    yearly_main_theme = random.choices(lst)
    yearly_departing_energies = random.choices(lst)
    yearly_talents = random.choices(lst)
    yearly_future_opportunities = random.choices(lst)
    yearly_obstacles = random.choices(lst)
    yearly_obstacles_deal = random.choices(lst)
    yearly_achieve = random.choices(lst)
    yearly = (
        yearly_main_theme,
        yearly_departing_energies,
        yearly_talents,
        yearly_future_opportunities,
        yearly_obstacles,
        yearly_obstacles_deal,
        yearly_achieve,
    )
    return yearly


def draw_career():
    """
    This function draws tarots cards from Lst, to get a tarot spread to predict career.
    Returns 3 cards, for stay/leave, new opportunity, and advice for career of user.
    https://mywanderingfool.com/tarot/tarot/best-tarot-card-spreads-career-finances/
    """
    career_stay_leave = random.choice(lst)
    career_new_oopportunity = random.choice(lst)
    career_advice = random.choice(lst)
    career = career_stay_leave, career_new_oopportunity, career_advice
    return career

"""
    From Text Analysis Project:

    model_engine = "davinci"
    temperature = 0.5
    max_tokens = 50
    stop_sequence = "."

    prompt = f"Act as if you are {musician}, and you're writing a song. What's a line from your song?:"
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.5,
    )

    sentence = response.choices[0].text.strip()
    return sentence
"""


def love_reading():
    tarot = draw_love()
    print(tarot)
    love_1 = tarot[0]
    love_2 = tarot[1]
    love_3 = tarot[2]
    prompt = f"Write a paragraph describing the meaning of the tarot card '{love_1}' in the context of past love experiences. Make sure you state what card it is and please be specific."

    # Make an API call to ChatGPT for generating the paragraph
    response1 = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7,
    )

    # Extract the generated text from the API response
    reply1 = response1.choices[0].text.strip()

    prompt = f"Write a paragraph describing the meaning of the tarot card '{love_2}' in the context of current love experiences. Make sure you state what card it is and please be specific."

    # Make an API call to ChatGPT for generating the paragraph
    response2 = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7,
    )

    # Extract the generated text from the API response
    reply2 = response2.choices[0].text.strip()

    prompt = f"Write a paragraph describing the meaning of the tarot card '{love_3}' in the context of past love experiences in under 200 words. Make sure you state what card it is and please be specific."

    # Make an API call to ChatGPT for generating the paragraph
    response3 = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0.7,
    )

    # Extract the generated text from the API response
    reply3 = response3.choices[0].text.strip()

    return f"\nPast love: \n{reply1}\nPresent love:\n{reply2}\nFuture love:\n{reply3}"

def yearly_forecast_reading():
    tarot = draw_yearly_forecast()
    print(tarot)
    yearly_1 = tarot[0]
    yearly_2 = tarot[1]
    yearly_3 = tarot[2]
    yearly_4=tarot[3]
    yearly_5=tarot[4]
    yearly_6=tarot[5]
    yearly_7=tarot[6]

    prompt = f"Write a paragraph describing the meaning of the tarot card '{yearly_1}' in the context of this year's main theme. Make sure you state what card it is and please be specific."

    # Make an API call to ChatGPT for generating the paragraph
    response1 = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7,
    )

    # Extract the generated text from the API response
    reply1 = response1.choices[0].text.strip()

    prompt = f"Write a paragraph describing the meaning of the tarot card '{yearly_2}' in the context of this year's departing energies. Make sure you state what card it is and please be specific."

    # Make an API call to ChatGPT for generating the paragraph
    response2 = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7,
    )

    # Extract the generated text from the API response
    reply2 = response2.choices[0].text.strip()

    prompt = f"Write a paragraph describing the meaning of the tarot card '{yearly_3}' in the context of this year's talents in under 200 words. Make sure you state what card it is and please be specific."

    # Make an API call to ChatGPT for generating the paragraph
    response3 = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0.7,
    )

    # Extract the generated text from the API response
    reply3 = response3.choices[0].text.strip()

    prompt = f"Write a paragraph describing the meaning of the tarot card '{yearly_4}' in the context of this year's opportunities in under 200 words. Make sure you state what card it is and please be specific."

    # Make an API call to ChatGPT for generating the paragraph
    response4 = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0.7,
    )

    # Extract the generated text from the API response
    reply4 = response4.choices[0].text.strip()

    prompt = f"Write a paragraph describing the meaning of the tarot card '{yearly_5}' in the context of this year's obstacles in under 200 words. Make sure you state what card it is and please be specific."

    # Make an API call to ChatGPT for generating the paragraph
    response5 = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0.7,
    )

    # Extract the generated text from the API response
    reply5 = response5.choices[0].text.strip()

    prompt = f"Write a paragraph describing the meaning of the tarot card '{yearly_6}' in the context of this year's ways to deal with obstacles in under 200 words. Make sure you state what card it is and please be specific."

    # Make an API call to ChatGPT for generating the paragraph
    response6 = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0.7,
    )

    # Extract the generated text from the API response
    reply6 = response6.choices[0].text.strip()

    prompt = f"Write a paragraph describing the meaning of the tarot card '{yearly_7}' in the context of this year's achievements in under 200 words. Make sure you state what card it is and please be specific."

    # Make an API call to ChatGPT for generating the paragraph
    response7 = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0.7,
    )

    # Extract the generated text from the API response
    reply7 = response7.choices[0].text.strip()

    return f"\nMain theme: \n{reply1}\nDeparting energies:\n{reply2}\nThis year's talents: \n{reply3} \nFuture opportunities:\n{reply4}\nObstacles:\n{reply5}\nHow to deal with obstacles:\n{reply6}\nThis year's achievements:\n{reply7}"

def career_reading():
    tarot = draw_career()
    print(tarot)
    career_1 = tarot[0]
    career_2 = tarot[1]
    career_3 = tarot[2]
    prompt = f"Write a paragraph describing the meaning of the tarot card '{career_1}' in the context of past career experiences. Make sure you state what card it is and please be specific."

    # Make an API call to ChatGPT for generating the paragraph
    response1 = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7,
    )

    # Extract the generated text from the API response
    reply1 = response1.choices[0].text.strip()

    prompt = f"Write a paragraph describing the meaning of the tarot card '{career_2}' in the context of current career experiences. Make sure you state what card it is and please be specific."

    # Make an API call to ChatGPT for generating the paragraph
    response2 = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7,
    )

    # Extract the generated text from the API response
    reply2 = response2.choices[0].text.strip()

    prompt = f"Write a paragraph describing the meaning of the tarot card '{career_3}' in the context of past career experiences in under 200 words. Make sure you state what card it is and please be specific."

    # Make an API call to ChatGPT for generating the paragraph
    response3 = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0.7,
    )

    # Extract the generated text from the API response
    reply3 = response3.choices[0].text.strip()

    return f"\nPast career: \n{reply1}\nPresent career:\n{reply2}\nFuture career:\n{reply3}"

def answer(type):
    if type=="love":
        return love_reading()
    if type=="yearly forecast":
        return yearly_forecast_reading()
    if type=="career":
        return career_reading()

def main():
    # print(draw_love())
    # print(draw_yearly_forecast())
    # print(draw_career())
    # print(draw_general())

    print(answer("love"))

if __name__ == "__main__":
    main()

import argparse
import os
import random

from openai import OpenAI

import pendulum
import requests
from dotenv import load_dotenv
from BingImageCreator import ImageGen

from quota import make_quota
from todoist import make_todoist

load_dotenv()

# required settings. config in github secrets
# -------------
# OpenAI: https://platform.openai.com/account/usage
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
# Telegram Bot Token
TG_BOT_TOKEN = os.environ['TG_BOT_TOKEN']
# Telegram Chat ID to want to send the message to
TG_CHAT_ID = os.environ['TG_CHAT_ID']
# Get Weather Information: https://github.com/baichengzhou/weather.api/blob/master/src/main/resources/citycode-2019-08-23.json to find the city code
# Shanghai 101020100
# Hangzhou 101210101 by default
WEATHER_CITY_CODE = os.environ.get('WEATHER_CITY_CODE', '101210101')
# -------------

# Optional Settings. config in github secrets.
# -------------
# 每日一句名人名言 - TIAN_API_KEY: https://www.tianapi.com/console/
# https://www.tianapi.com/console/
TIAN_API_KEY = os.environ.get('TIAN_API_KEY', '')
# Bing Cookie if image to be generated from Dalle3. Leave empty to use OpenAI by default
BING_COOKIE = os.environ.get('BING_COOKIE', '')
# 每日待办事项 todoist
TODOIST_API = os.environ.get('TODOIST_API', '')
# -------------

# Message list
MESSAGES = ['又到了新的一天了！']


# get today's weather
# city hard coded in API URL. You may change it based on city code list below
def make_weather(city_code):
    print(f'Start making weather...')
    WEATHER_API = f'http://t.weather.sojson.com/api/weather/city/{city_code}'
    # https://github.com/baichengzhou/weather.api/blob/master/src/main/resources/citycode-2019-08-23.json to find the city code
    DEFAULT_WEATHER = "未查询到天气，好可惜啊"
    WEATHER_TEMPLATE = "今天是{date} {week}，{city}的天气是{type}，{high}，{low}，空气质量指数{aqi}"

    try:
        r = requests.get(WEATHER_API)
        if r.ok:
            weather = WEATHER_TEMPLATE.format(
                date=r.json().get("data").get("forecast")[0].get("ymd"), week=r.json().get("data").get("forecast")[0].get("week"),
                city=r.json().get("cityInfo").get("city"),
                type=r.json().get("data").get("forecast")[0].get("type"), high=r.json().get("data").get("forecast")[0].get("high"),
                low=r.json().get("data").get("forecast")[0].get("low"), aqi=r.json().get("data").get("forecast")[0].get("aqi")
            )
            return weather
        return DEFAULT_WEATHER
    except Exception as e:
        print(type(e), e)
        return DEFAULT_WEATHER

# get random poem
# return sentence(used for make pic) and poem(sentence with author and origin)


def get_poem():
    SENTENCE_API = "https://v1.jinrishici.com/all"
    DEFAULT_SENTENCE = "落日净残阳 雾水拈薄浪 "
    DEFAULT_POEM = "落日净残阳，雾水拈薄浪。 —— Xiaowen.Z / 卜算子"
    POEM_TEMPLATE = "{sentence} —— {author} / {origin}"

    try:
        r = requests.get(SENTENCE_API)
        if r.ok:
            sentence = r.json().get("content")
            poem = POEM_TEMPLATE.format(
                sentence=sentence, author=r.json().get("author"), origin=r.json().get("origin")
            )
            return sentence, poem
        return DEFAULT_SENTENCE, DEFAULT_POEM
    except Exception as e:
        print(type(e), e)
        return DEFAULT_SENTENCE, DEFAULT_POEM


# create pic
# return url, the image will not be save to local environment
def make_pic_from_openai(sentence):
    """
    return the link formd
    """
    # openai.api_key = OPENAI_API_KEY
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=OPENAI_API_KEY,
    )
    print(f'calling open ai for image creation...')
    response = client.images.generate(
        prompt=sentence, n=1, size="1024x1024", model="dall-e-3", style="vivid")

    image_url = response.data[0].url
    print(f'image_url:{image_url}')
    print(f'image_revised_prompt: {response.data[0].revised_prompt}')
    print(f'full response: {response}')
    # s = requests.session()
    # index = 0
    # while os.path.exists(os.path.join(new_path, f"{index}.jpeg")):
    #     index += 1
    # with s.get(image_url, stream=True) as response:
    #     # save response to file
    #     response.raise_for_status()
    #     with open(os.path.join(new_path, f"{index}.jpeg"), "wb") as output_file:
    #         for chunk in response.iter_content(chunk_size=8192):
    #             output_file.write(chunk)

    return image_url, "Image Powered by OpenAI DELL.E-3"

# create pic from bing image generator
# once Dalle3 api is available, this might be retired.


def make_pic_from_bing(sentence, bing_cookie):
    # for bing image when dall-e3 open drop this function
    i = ImageGen(bing_cookie)
    images = i.get_images(sentence)
    return images, "Image Powered by Bing DALL.E-3"

# try Dalle-3 from Bing first, then OpenAI Image API
def make_pic(sentence):
    if BING_COOKIE is not None and BING_COOKIE != '':
        try:
            images, image_comment = make_pic_from_bing(sentence, BING_COOKIE)
            return images[0], image_comment
        except Exception as e:
            print(f'Image generated from Bing failed: {type(e)}')
            print(type(e), e)
    else:
        print('Bing Cookie is not set. Use OpenAI to generate Image')
    image_url, image_comment = make_pic_from_openai(sentence)
    return image_url, image_comment


def make_poem():
    print(f'Start making poem...')
    sentence, poem = get_poem()
    sentence_processed = sentence.replace(
        "，", " ").replace("。", " ").replace(".", " ")
    print(f'Processed Sentence: {sentence_processed}')
    image_url, image_comment = make_pic(sentence_processed)
    poem_message = f'今日诗词和配图：{poem}\r\n\r\n{image_comment}'

    return image_url, poem_message

# send message to telegram
# send image with caption if the image arg is not None


def send_tg_message(tg_bot_token, tg_chat_id, message, image=None):
    print(f'Sending to Chat {tg_chat_id}')
    if image is None:
        try:
            request_url = "https://api.telegram.org/bot{tg_bot_token}/sendMessage".format(
                tg_bot_token=tg_bot_token)
            request_data = {'chat_id': tg_chat_id, 'text': message}
            response = requests.post(request_url, data=request_data)
            return response.json()
        except Exception as e:
            print("Failed sending message to Telegram Bot.")
            print(type(e), e)
            return ""
    else:
        try:
            photo_url = image
            request_url = "https://api.telegram.org/bot{tg_bot_token}/sendPhoto".format(
                tg_bot_token=tg_bot_token)
            request_data = {'chat_id': tg_chat_id,
                            'photo': photo_url, 'caption': message}
            response = requests.post(request_url, data=request_data)
            return response.json()
        except Exception as e:
            print("Failed sending message to Telegram Bot with image.")
            print(type(e), e)
            return ""

# generate content from list of messages


def make_message(messages):
    message = "\r\n---\r\n".join(messages)
    return message

# generate content
# send to tg


def main():
    print("Main started...")
    # default process the poem, image and weather.
    MESSAGES.append(make_weather(WEATHER_CITY_CODE))
    image_url, poem_message = make_poem()
    MESSAGES.append(poem_message)

    # --------
    # Optional process - Daily Quota
    if TIAN_API_KEY is not None and TIAN_API_KEY != '':
        MESSAGES.append(make_quota(TIAN_API_KEY))
    # --------
    # --------
    # Optional process - 每日待办事项 todoist
    if TODOIST_API is not None and TODOIST_API != '':
        MESSAGES.append(make_todoist(TODOIST_API))
    # --------

    # Build full content and send to TG
    full_message = make_message(MESSAGES)
    print("Message constructed...")
    print()
    print("Sending to Telegram...")
    r_json = send_tg_message(tg_bot_token=TG_BOT_TOKEN,
                             tg_chat_id=TG_CHAT_ID, message=full_message, image=image_url)
    print(r_json)


if __name__ == "__main__":
    main()

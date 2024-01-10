import os

import openai
import logging

# google search 기능
import json
import requests

from dotenv  import load_dotenv






load_dotenv()  # .env 파일에서 환경변수를 불러옵니다.

CHATGPT_ENGINE = os.getenv("CHATGPT_ENGINE", "gpt-3.5-turbo-0613")
BOT_PENCIL_ICON = os.getenv("BOT_PENCIL_ICON", "✍")

UPDATE_CHAR_RATE = 5

openai.api_key = os.environ["OPENAI_API_KEY"]
Google_SEARCH_ENGINE_ID = os.environ["Google_SEARCH_ENGINE_ID"]
Google_API_KEY = os.environ["Google_API_KEY"]

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def chatgpt_response(messages):
    try:
        response = openai.ChatCompletion.create(
            model=CHATGPT_ENGINE,
            messages=messages
        )
        return response.choices[0].message.content
    except (KeyError, IndexError) as e:
        return "GPT3 Error: " + str(e)
    except Exception as e:
        return "GPT3 Unknown Error: " + str(e)


async def chatgpt_callback_response(messages, functions):

    content = ""

    try:

        response = openai.ChatCompletion.create(
            model=CHATGPT_ENGINE,
            messages=messages,
            functions=functions,
            function_call="auto",
        )

        response_message = response["choices"][0]["message"]

        return response_message

        


    except (KeyError, IndexError) as e:
        return "GPT3 Error: " + str(e)
    except Exception as e:
        logger.error("GPT3 Unknown Error: " + str(e))
        if content:
            return content
        return "GPT3 Unknown Error: " + str(e)
    






start_page = 1 # 몇 페이지를 검색할 것인지. 한 페이지 당 10개의 게시물을 받아들일 수 있습니다.



def get_search_info(keyword):
    
    url = f"https://www.googleapis.com/customsearch/v1?key={Google_API_KEY}&cx={Google_SEARCH_ENGINE_ID}&q={keyword}&start={start_page}"
    response = requests.get(url).json()
    
    search_result = response.get("items")
    
    search_info = {}
        
    search_info['link'] = search_result[0]['link'] 
    search_info['title'] = search_result[0]['title'] 
    search_info['snippet'] = search_result[0]['snippet']
    
    return json.dumps(search_info)
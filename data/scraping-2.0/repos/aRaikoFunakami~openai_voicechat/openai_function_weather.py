import json
import os, logging
from typing import Any

import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import requests

"""
langchainのConversationalRetrievalChain.from_llmを利用する場合にはgpt-4でないと良い回答が得られない
"""
model_name = "gpt-3.5-turbo-0613"
# model_name = "gpt-4-0613"

# default_persist_directory = "./chroma_split_documents"
default_persist_directory = "./chroma_load_and_split"


# load config
def load_config():
    config_file = os.path.dirname(__file__) + "/config.json"
    config = None
    with open(config_file, "r") as file:
        config = json.load(file)
    return config


#
# call by openai functional calling
#
def get_weather_info(latitude, longitude):
    base_url = "https://api.open-meteo.com/v1/forecast"
    parameters = {
        "latitude": latitude,
        "longitude": longitude,
        #        "current_weather": "true",
        "hourly": "temperature_2m,relativehumidity_2m",
        "timezone": "Asia/Tokyo",
    }
    response = requests.get(base_url, params=parameters)
    if response.status_code == 200:
        data = response.json()
        logging.info(data)
        return json.dumps(data)
    else:
        return None


#
# call by openai functional calling
#
weather_function = {
    "name": "get_weather_info",
    "description": "Get current weather from latitude and longitude information",
    "parameters": {
        "type": "object",
        "properties": {
            "latitude": {
                "type": "string",
                "description": "latitude",
            },
            "longitude": {
                "type": "string",
                "description": "longitude",
            },
        },
        "required": ["latitude", "longitude"],
    },
}
#
#
# Test codes: Verify that the registered function call is called as expected
#
#


def call_defined_function(message):
    function_name = message["function_call"]["name"]
    logging.debug("選択された関数を呼び出す: %s", function_name)
    arguments = json.loads(message["function_call"]["arguments"])
    if function_name == "get_weather_info":
        return get_weather_info(
            latitude=arguments.get("latitude"),
            longitude=arguments.get("longitude"),
        )
    else:
        return None


def non_streaming_chat(text):
    # 関数と引数を決定する
    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "user", "content": text}],
            functions=[weather_function],
            function_call="auto",
        )
    except openai.error.OpenAIError as e:
        error_string = f"An error occurred: {e}"
        print(error_string)
        return {"response": error_string, "finish_reason": "stop"}

    message = response["choices"][0]["message"]
    logging.debug("message: %s", message)
    # 選択した関数を実行する
    if message.get("function_call"):
        function_response = call_defined_function(message)
        #
        # Returns the name of the function called for unit test
        #
        return message["function_call"]["name"]
    else:
        return "chatgpt"


template = """
条件:
- 50文字以内で回答せよ

入力文:
{}
"""


def chat(text):
    logging.debug(f"chatstart:{text}")
    config = load_config()
    openai.api_key = config["openai_api_key"]
    q = template.format(text)
    return non_streaming_chat(q)


queries = [
    ["今日の東京の天気はどうですか？", "get_weather_info"],
    ["明日の大阪の天気を教えてください。", "get_weather_info"],
    ["週末の福岡の天気予報を知りたいです。", "get_weather_info"],
    ["来週の水曜日に札幌で雨が降る予報はありますか？", "get_weather_info"],
    ["今日の夜、名古屋で気温はどれくらいですか？", "get_weather_info"],
    ["What is the weather like in Tokyo today?", "get_weather_info"],
    ["Can you tell me the weather in Osaka tomorrow?", "get_weather_info"],
    [
        "I would like to know the weather forecast for Fukuoka this weekend.",
        "get_weather_info",
    ],
    ["Will it rain in Sapporo next Wednesday?", "get_weather_info"],
    ["What is the temperature in Nagoya tonight?", "get_weather_info"],
]


def main():
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(filename)s:%(funcName)s[%(lineno)d] - %(message)s",
    )
    for query in queries:
        response = chat(query[0])
        print(f"[{query[1] == response}] 期待:{query[1]}, 実際:{response}, 質問:{query[0]}")


if __name__ == "__main__":
    main()

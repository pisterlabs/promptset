import json
import os

import requests
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'API key not found')

gpt_4_vision = "gpt-4-vision-preview"
gpt_3_5_turbo = "gpt-3.5-turbo"

chain = ChatOpenAI(model=gpt_3_5_turbo,
                   max_tokens=1024,
                   openai_api_key=OPENAI_API_KEY)

prompt_get_words = "Go through the text line by line and return only the words with the frame around them. " \
                   "This frame looks like drawn by hand with a gray pencil. " \
                   "There should be multiple words marked with such frame. " \
                   "If you are not sure about any of the words return it anyway. " \
                   "Ignore all the words with other color, font and size. " \
                   "Return the list of words in format [word1, word2, word3]"

prompt_format_words = "You are my flashcards creator assistant. " \
                      "I give you a list of words and you go through them and process them to return me each word as: " \
                      "- singular (if applicable) " \
                      "- present indicative (if applicable) " \
                      "- with correct article (if applicable)\n\n" \
                      "Return JSON without any description" \
                      "for example:\n" \
                      " - user: [casas, hablábamos, íbamos, comí, gatos]\n" \
                      " - return: [\"la casa\", \"hablar\", \"ir\", \"comer\", \"el gato\"\n\n" \
                      "Return only the words without any explanations."

prompt_create_flashcards = "You are my flashcards creator assistant. I give you a list of words in Spanish and " \
                           "you go through them to translate them to Polish. Besides that, you also give me a very short " \
                           "and simple example in present tense of a sentence with that word in Spanish. Return the response in the form of JSON.\n\n" \
                           "For example:\n" \
                           "User: ['la casa', 'el huevo']\n\n" \
                           "Response:\n" \
                           "\"flashcards\": [\n" \
                           "    {\n" \
                           "        \"spanish\": 'la casa',\n" \
                           "        \"polish\": \"dom\",\n" \
                           "        \"example\": \"Esta [[ dom ]] es bonita\"\n" \
                           "    },\n" \
                           "    {\n" \
                           "        \"spanish\": \"el huevo\",\n" \
                           "        \"polish\": \"jajko\",\n" \
                           "        \"example\": \"A mi me gusta [[ jajko ]]\"\n" \
                           "    }\n" \
                           "]"


def create_flashcard(formatted_words):
    messages = [
        SystemMessage(content=f"{prompt_create_flashcards}"),
        HumanMessage(content=f"{formatted_words}")
    ]
    res = chain(messages)
    flashcards = json.loads(res.content)['flashcards']
    print(flashcards)
    return flashcards


def format_extracted_words(words):
    messages = [
        SystemMessage(content=f"{prompt_format_words}"),
        HumanMessage(content=f"{words}"),
    ]
    res = chain(messages)
    formatted_words = json.loads(res.content)
    print(formatted_words)
    return formatted_words


def extract_words_from_file(file):
    if OPENAI_API_KEY == 'API key not found':
        raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{prompt_get_words}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{file}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response

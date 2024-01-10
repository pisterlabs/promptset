import openai
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

MODEL = "gpt-3.5-turbo"


def GPT_response(system_message, user_message, max_tokens=1300, temperature=0.8):
    response = openai.ChatCompletion.create(
        model=MODEL,
        max_tokens=max_tokens,
        temperature= temperature,
        messages=[{"role": "system", "content": system_message},
                  {"role": "user", "content": user_message}
                  ],
    )
    return response

def rich_character(prompt, story, name, max_tokens=1300, temperature=0.8):
    system_prompt = prompt + story
    user_prompt = "請豐富" + name + "的個人資訊。"
    response = GPT_response(system_prompt, user_prompt, max_tokens=max_tokens, temperature=temperature)
    # print(response.choices[0]['message']['content'])
    return response.choices[0]['message']['content']

def character(name, prompt, story, character_info, query, max_tokens=1300, temperature=0.8):
    character_name = "現在你需要假扮的嫌疑人為" + name + "我將給你故事內容和嫌疑人的資料"

    prompt = prompt + character_name + story + character_info

    response = GPT_response(prompt, query, max_tokens=max_tokens, temperature=temperature)
    # print(response.choices[0]['message']['content'])
    return response.choices[0]['message']['content']

    
def rich_place(prompt, story, max_tokens=1300, temperature=0.8):
    system_prompt = prompt 
    user_prompt = "我將給你一段偵探故事，請你幫我豐富案發現場的資訊。" + story
    response = GPT_response(system_prompt, user_prompt, max_tokens=max_tokens, temperature=temperature)
    # print(response.choices[0]['message']['content'])
    return response.choices[0]['message']['content']

def summary(prompt, story, max_tokens=1300, temperature=0.8):
    system_prompt = prompt 
    user_prompt = "我將給你一段偵探故事，請你幫我進行總結。" + story
    response = GPT_response(system_prompt, user_prompt, max_tokens=max_tokens, temperature=temperature)
    # print(response.choices[0]['message']['content'])
    return response.choices[0]['message']['content']


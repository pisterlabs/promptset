import os
from dotenv import load_dotenv
import openai
from .tts_config import create_text_to_speech
from . import global_state


def AI_chat_GPT(voice_input):
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    # マイクから取ってきたものを関数化
    user_input = voice_input
    if user_input:
        print(f"AI_selected{global_state.ai_selection}")
        print(f"Language_selected{global_state.language_selection}")
        print(f"Mode_selected{global_state.mode_selection}")
        print(f"You:{user_input}")
    else: 
        print("user_inputを受け取っていません")
    # aiの設定
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo-1106",
        messages = [{"role": "system", "content": "You are a friendly language teacher. Your response must be less than 50 words."},
                    {"role": "system", "content": user_input}
        ],
        max_tokens=100
    ) 
    # aiから生成されたものから文字部分のみ抽出
    gpt_response = response.choices[0].message.content
    # gpt_response = "Hello World"
    print(f"AI:{gpt_response}") 
    text_to_speech_data = create_text_to_speech(gpt_response)
    
    return gpt_response, text_to_speech_data

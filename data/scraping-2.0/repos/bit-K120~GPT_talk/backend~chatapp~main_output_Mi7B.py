import os
from dotenv import load_dotenv
import openai
from .tts_config import create_text_to_speech


def AI_chat_Mi7B(voice_input):
    load_dotenv()
    PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
    # マイクから取ってきたものを関数化
    user_input = voice_input
    if user_input:
        print(f"AI_selected{global_state.ai_selection}")
        print(f"Language_selected{global_state.language_selection}")
        print(f"Mode_selected{global_state.mode_selection}")
        print(f"You:{user_input}")
    else: 
        print("user_inputを受け取っていません")    
            # perplexityのメッセージ設定
    messages = [
        {
        "role": "system",
        "content": (
            "You are a friendly English teacher, and you need to "
            "engage in a friendly conversation with user."
            "Your response has to be less than 15 words."


        ),
        },
        {
        "role": "user",
        "content": (user_input),
        },
        ]
        # aiの設定

    response = openai.ChatCompletion.create(
        model="mistral-7b-instruct",
        messages=messages,
        api_base="https://api.perplexity.ai",
        api_key=PERPLEXITY_API_KEY,   
        )                                           
                      
        # aiから生成されたものから文字部分のみ抽出
    gpt_response = response.choices[0].message.content
    print(f"AI:{gpt_response}")
    text_to_speech_data = create_text_to_speech(gpt_response)
    
    return gpt_response, text_to_speech_data

import openai 
from config import OPENAI_API_KEY,MAX_ANS


def chat_answer(input_text):
    openai.api_key= OPENAI_API_KEY
    input_text = input_text +"という悩みに対し," + "保育士として，" + str(MAX_ANS)+"文字で，親切に寄り添いながら悩みを聞き，最後は疑問形で終わらせてください．"
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "保育士"},
            {"role": "user", "content": input_text}
        ]
    )
    text_unicode = response["choices"][0]["message"]["content"]
    OUTPUT_TEXT = "AI保育士:"+text_unicode
    # print(text_unicode)
    return OUTPUT_TEXT

# # デバック用
# while True :
#     input_text = input("あなたの悩みを入力してください：")
#     if input_text == "exit":
#         break
#     print(chat_answer(input_text))
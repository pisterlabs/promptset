import openai
from dotenv import load_dotenv

load_dotenv()

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "user",
            "content": "情報の正確性という観点から次のツイートを分析してください。\n"
                       + "ツイート：50％以上の科学者は気候変動を信じていない。"
        }
    ],
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)
print(response["choices"][0]["message"]["content"])

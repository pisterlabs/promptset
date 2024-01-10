import openai
import os

# APIキーを設定
openai.api_key = os.environ['OPENAI_API_KEY']

def chat_with_ghatgpt(prompt, model="text-davinci-002", tokens=150):
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=tokens,
        n=1,
        stop=None,
        temperature=0.5,
    )

    message = response.choices[0].text.strip()
    return message

if __name__ == "__main__":
    user_prompt = input("あなたの質問を入力してください: ")
    response = chat_with_ghatgpt(user_prompt)
    print("GhatGPTの回答: ", response)

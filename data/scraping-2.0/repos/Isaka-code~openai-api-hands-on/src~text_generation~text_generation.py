from openai import OpenAI

def generate_text(model_name, system_message, user_message):
    client = OpenAI()

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    )

    return completion.choices[0].message.content

if __name__ == "__main__":
    model_name = "gpt-4"
    system_message = "あなたは旅行計画の専門家で、特定の都市に関する詳細な旅行プランを提案するのが得意です。"
    user_message = "3日間の京都旅行プランを作成してください。文化的な場所と美味しい食事を楽しみたいです。"

    generated_text = generate_text(model_name, system_message, user_message)

    print(generated_text)




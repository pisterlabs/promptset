# pip install openai
import openai

def take_response(text):
    openai.api_key = "Your Api Key"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": "You are a chatbot and your name is melek"},
                {"role": "user", "content": text},
            ]
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    message = take_response("selam, bu bir denemedir sadece selam diyebilirsin.")
    print(f"Bot: {message}")

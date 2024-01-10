import openai
import os

openai_api_key = os.environ["PERSONALITYGPT_KEY"]

client = openai.OpenAI()


def get_completion(context, user_input, model = "gpt-4-1106-preview"):
    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": user_input}
    ]

    response = client.chat.completions.create(
        model = model,
        messages = messages,
        temperature = 1.5
    )

    return response; 

def main():
    print(get_completion("You are a cool assistant", "Give me 2 ways, in bullet points, to be cool"))

if __name__ == "__main__":
    main()
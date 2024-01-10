import openai

with open('openai_session.txt', 'r') as file:
    api_key = file.read().strip()

openai.api_key = api_key


def ask_gpt3_turbo(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

def main():
    print("OpenAI GPT-3.5-turbo 챗봇")
    print("종료하려면 'exit'를 입력하세요.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        prompt = f"You: {user_input}\nBot:"
        response = ask_gpt3_turbo(prompt)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()
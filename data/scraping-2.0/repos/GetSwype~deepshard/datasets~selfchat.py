import openai
import dotenv
import os

dotenv.load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]

pre_prompt = "Your goal is to maintain a conversation with the other AI model while keeping the conversation informative and engaging."

def chat_with_model(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"{pre_prompt}\n\n{prompt}",
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )

    return response.choices[0].text.strip()

def main():
    conversation_history = []

    for i in range(10):  # Adjust the number of conversation turns as needed
        if i % 2 == 0:
            prompt = f"AI 1: {conversation_history[-1]}" if conversation_history else "AI 1: Hi, AI 2! How are you today?"
            response = chat_with_model(prompt)
            conversation_history.append(f"AI 2: {response}")
        else:
            prompt = f"AI 2: {conversation_history[-1]}"
            response = chat_with_model(prompt)
            conversation_history.append(f"AI 1: {response}")

    for line in conversation_history:
        print(line)

if __name__ == "__main__":
    main()

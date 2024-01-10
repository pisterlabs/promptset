import openai
import json

def load_role_from_file(filename):
    with open(filename, 'r') as file:
        return file.read().strip()

def load_api_key_from_file(filename):
    with open(filename, 'r') as file:
        return file.read().strip()

def chat_with_gpt3(prompt, role):
    openai.api_key = load_api_key_from_file("openaiapikey.txt")
    model = "gpt-3.5-turbo"
    messages = [{"role": role, "content": prompt}]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        n=1,
        temperature=0.8,
    )

    return response.choices[0].message["content"]

def main():
    prompt_agent = load_role_from_file("promptagent.txt")
    prompt_monitor = load_role_from_file("promptmonitor.txt")
    conversation_history = []

    for i in range(10):
        if i % 2 == 0:
            role = "user"
            prompt = prompt_agent
        else:
            role = "assistant"
            prompt = prompt_monitor

        response = chat_with_gpt3(prompt, role)
        conversation_history.append({"role": role, "content": response})

        print(f"{role}: {response}")

if __name__ == "__main__":
    main()

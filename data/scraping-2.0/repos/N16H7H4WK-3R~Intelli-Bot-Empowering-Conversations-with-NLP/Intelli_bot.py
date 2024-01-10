from openai import ChatCompletion

# API key
api_key = "PLACE_YOUR_OPENAI_API_KEY_HERE"
# Variable to store the chat log
chat_file = None

def read_information(file_path):
    with open(file_path, "r") as file:
        return file.read()

def chat(question):
    global chat_file
    if chat_file is not None:
        chat_file += f"User: {question}\n"
    else:
        file_path = "data_trained.txt"
        information = read_information(file_path)
        chat_file = f"{information}\n"
        chat_file += f"User: {question}\n"

    response = ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": chat_file}],
        temperature=0.85,
        max_tokens=100,
        n=1,
        stop=None,
        api_key=api_key,
    )
    answer = response.choices[0].message.content.strip()
    chat_file += f"\n{answer}\n"
    return answer

import os
import openai

openai.api_key=os.environ['OPENAI_API_KEY']

def create_doc():
    system_message = "You are a professional technical documentation writer. The user will provide text from which you should create a documentation"
    model = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "system", "content": system_message}, {"role": "user", "content": _read_text()}])
    response = model['choices'][0]['message']['content']
    _write(response)

def _write(text):
    with open("doc.txt", "w") as file:
        file.write(text)

def _read_text():
    current_path = os.getcwd()
    file_name = "text.txt"
    file_path = os.path.join(current_path, file_name)
    with open(file_path, "r") as file:
        return file.read()

if __name__ == "__main__":
    create_doc()
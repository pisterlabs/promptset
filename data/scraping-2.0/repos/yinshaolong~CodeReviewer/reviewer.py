
import openai
from dotenv import load_dotenv
import os
import argparse

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()

model = {3: "gpt-3.5-turbo", 4: "gpt-4-1106-preview"}

def parse_args():
    parser = argparse.ArgumentParser(description="Simple code reviwer for a file")
    parser.add_argument("-m", default = '3',type=str, help="determines the gpt model the user would like to use")
    parser.add_argument("-f", default = "failing_code.js", type=str, help="determines the file the user would like to review")
    parser.add_argument("-p", default = False, type=bool, help="determines the prompt the user would like to use")
    return parser

def get_file_contents(file):
    prompt_string = ['"""']
    with open(file, "r") as f:
        prompt_string.append(f.read())
        if parse_args().p != False:
            print("in parse")
            prompt_string.append("""The generated code should be a single list of dictionaries. The key to each dictionary is the original line of code. The value is the generated line of code. e.g. [{'add(a, b):\n   return a *b': 'add(operand1, operand2):\n   return operand1 + operand2'}], [{'    console.log(add("3", "four"))}': '    console.log(addNumbers(parseInt("3"), parseInt("4")));'}]""")
    prompt_string.append('"""')
    return "".join(prompt_string)

def get_prompt(file="prompt.txt"):
    return get_file_contents(file).replace("\n", "")

def chat(prompt = get_prompt()):
    args = parse_args()
    for data in client.chat.completions.create(
    model = model[int(args.m)],
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Code review the following file: {get_file_contents(args.f)}"}
    ],
    stream = True,
    ):
        yield data.choices[0].delta.content

def generate_code_review():
    reply = chat()
    messages = []
    with open("review.txt", "w") as f:
        f.write('')
    file_name = get_file_details()[0]
    for message in reply:
        print(message, end="", flush=True)
        with open(f"{file_name}_critique.txt", "a") as f:
            f.write(message) if message != None else f.write("")
            messages.append(message) if message != None else f.write("")
    return "".join(messages)

def get_file_details(file = parse_args().f):
    file_path = os.path.abspath(file)
    return os.path.splitext(file_path)

def generate_reviewed_code(messages):
    args = parse_args()
    #gets first and last index
    opening_index = messages.index("```")
    closing_index = messages.rfind("```")
    reviewed_code = messages[opening_index + 3:closing_index]
    #removes the first line of the code, in this case the codes respective language
    reviewed_code = reviewed_code[reviewed_code.index("\n") + 1:]
    print("reviewed code: ",reviewed_code)
    #gets the matching file extension so that it can be used to create a new file
    file_name, file_extension = get_file_details()
    with open(f"{file_name}_reviewed{file_extension}", "w") as f:
        f.write(reviewed_code)


def main():
    messages = generate_code_review()
    generate_reviewed_code(messages)

if __name__ == "__main__":
    main()

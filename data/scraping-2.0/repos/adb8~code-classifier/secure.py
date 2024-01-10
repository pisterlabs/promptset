import os
import openai
from dotenv import load_dotenv
import topics
import random
from prepare import remove_comments, remove_certain_chars, remove_extra_space, add_row_to_csv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_MINE")
openai.api_key = OPENAI_API_KEY

topics = topics.topics
max_tokens = 500
def generate_prompt(topic):
    prompt = """Provide me with a unique code snippet that demonstrates good input validation in Python. The program should be whole and perform a useful or entertaining task. The program should do more than simply regurgitate the input. The program should """ + topic + """. Ensure that the input is the right type of data, has an appropriate length, is within an appropriate range of values, will not cause errors, is in the right case, and meets a specified format. Do not add any comments inside the code snippet. Provide me with only the code snippet. Make the program sophisticated and intermediate level. Make the program a minimum of 100 lines."""
    return prompt

def request_snippet():
    result = openai.Completion.create(
        model="text-davinci-003",
        prompt=generate_prompt(random.choice(topics)),
        temperature=0.6,
        max_tokens=max_tokens,
    ).choices[0].text

    code_snippet = remove_comments(result)
    code_snippet = remove_certain_chars(code_snippet)
    code_snippet = remove_extra_space(code_snippet)
    return code_snippet

def append_snippet(snippet, label):
    data_to_add = [label, snippet]
    data_path = "snippets/secure.csv"
    add_row_to_csv(data_path, data_to_add)

def main():
    try:
        label = "Secure"
        snippet = request_snippet()
        append_snippet(snippet, label)
        print("Successfully appended data")
        main()
    except Exception as e:
        print(e)
        main()

if __name__ == "__main__":
    main()
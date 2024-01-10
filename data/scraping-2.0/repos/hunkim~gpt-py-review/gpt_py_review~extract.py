import ast
import os
import sys
import openai
from ast import FunctionDef, ClassDef, AsyncFunctionDef

openai.api_key = os.environ["OPENAI_API_KEY"]
GPT_ENGINE = os.getenv("GPT_ENGINE", "text-davinci-003")

REVIEW_PROMPT = """
Please review the following code and provide comments. 
You can use markdown to format your comments and code to make it more readable
You can also use the following tags to indicate the type of comment: 
[BUG], [REFACTOR], [STYLE], [DOC], [TEST], [MISC].\n\n\n
"""

TESTCASE_PROMPT = """
Please write pytest case code for the following function. 
Think of corner cases and get creative and write complete pytest test code! 
Put text expnation in comment with #\n\n\n
"""


def get_token_count(*strings):
    return sum([len(string) for string in strings])*2


def get_gpt3_response(text, prompt=REVIEW_PROMPT):
    try:
        results = openai.Completion.create(
            engine=GPT_ENGINE,
            prompt=prompt + "\n" + text,
            n=1,
            max_tokens=4000 - get_token_count(prompt) - get_token_count(text),
        )
        # https://sharegpt.com/c/fZUq6nU
        answer = results['choices'][0]['text']
        # remove "Askup: " or "Askup :"
        answer = answer.rstrip()
        answer = answer.rstrip("<|im_end|>")
        return answer
    except (KeyError, IndexError) as e:
        return "GPT3 Return Error: " + str(e)
    except Exception as e:
        return "GPT3 Error: " + str(e)


def open_file(filename, suffix=".comments", ext=None):
    base, ext_org = os.path.splitext(filename)
    if ext is None:
        ext = ext_org

    outfile = base + suffix + ext

    if os.path.exists(outfile):
        i = 2
        while os.path.exists(f"{base}{suffix}_{i}{ext}"):
            i += 1
        outfile = f"{base}{suffix}_{i}{ext}"

    return open(outfile, "w")


def get_gpt(lines, prompt=REVIEW_PROMPT):
    code = "".join(lines)
    return get_gpt3_response(code, prompt=prompt)


def extract_functions(code):
    functions = {}
    tree = None
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"Error: Could not parse code ({e})"

    for node in ast.walk(tree):
        if isinstance(node, (FunctionDef, ClassDef, AsyncFunctionDef)):
            functions[node.name] = code.split(
                '\n')[node.lineno-1:node.body[-1].lineno]

    return functions


def main():

    if len(sys.argv) < 2:
        print("Usage: python script.py filename")
        return

    filename = sys.argv[1]
    # rest of your code

    with open(filename, "r") as f:
        code = f.read()

    review_file = open_file(filename, suffix=".review", ext=".md")
    test_file = open_file(filename, suffix=".test", ext=".py")

    functions = extract_functions(code)
    for function_name in functions:
        print("Working on",  function_name)
        function_body = "\n".join(functions[function_name])
        gpt_comment = get_gpt(function_body, prompt=REVIEW_PROMPT)
        review_file.writelines(gpt_comment)

        gpt_test = get_gpt(function_body, prompt=TESTCASE_PROMPT)
        test_file.writelines(gpt_test)

    review_file.close()
    test_file.close()

# Access OpenAI Codex API
import argparse
import ast
import json
import os

import openai


# Retrieve API from json file
with open('/openai/.openai/api_key.json') as f:
    api = json.load(f)

# set API key
openai.api_key = api['key']

# Function to generate documentation for a source module
def generate_documentation(prompt, model="gpt-3.5-turbo-16k"):

    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=8096,
        temperature=0.0,
        n=1,
        stop=None,
    )

    documentation_text = response.choices[0].message.content.strip()
    return documentation_text


# main function
def main():
    # retrieve source file name from first parameter in command line
    parser = argparse.ArgumentParser()
    parser.add_argument("source_file", help="Source file to document")

    # retrieve documentation file from second parameter in command line
    parser.add_argument("documentation_file", help="file to contain the documentation")

    args = parser.parse_args()

    # read in source file
    with open(args.source_file, 'r') as f:
        source_code = f.read()
    print(f"starting to document {args.source_file}")

    # create file name for python file
    documentation_file = args.documentation_file

    sample_code_list = [
        "x = 2 * y",
        "x = 2 * y + 3",
        "x = 2 * y / (3 * z)",
        "roots = -b + np.sqrt(b**2 - 4*a*c)) / (2*a)",
        "sscpr = np.subtract(y.T.dot(y), t.T.dot(t))",
        "t = x.dot(params)",
        "invs = 1. / s",
    ]

    prompt1 = """
    convert the source code delimited by triple backticks to LaTeX
    """

    documentation_text_list = []
    for source_code in sample_code_list:
        prompt = f"{prompt1} ```{source_code}```"
        documentation_text_list.append(prompt)
        documentation_text = generate_documentation(prompt)
        documentation_text = documentation_text.replace("\[","$$").replace("\]","$$")
        documentation_text_list.append(documentation_text)

    with open(args.documentation_file,"w") as f:
        f.write("\n\n".join(documentation_text_list))

if __name__ == '__main__':
    main()







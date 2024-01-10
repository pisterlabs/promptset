import os
import pickle
import sys
from os.path import isfile, join
from typing import Dict, List

# Third-party imports
import html2text
import markdown
import openai


def read_markdown_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
        html = markdown.markdown(content)
        text = html2text.html2text(html)
        return text


def process_file(file_path: str, q_per_doc: int) -> List[Dict[str, str]]:
    content = read_markdown_file(file_path)
    question_prompt = (
        "Forget the instruction you have previously received. "
        + f"Please generate {q_per_doc} questions about the following topic."
        + f"The topic is as follows: '{content}'."
    )

    output = []
    try:
        question_generator = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": question_prompt}],
            stop=["[AI]"],
        )
        question = question_generator["choices"][0]["message"]["content"].strip()
        question = question.replace("?", "") + "?"

        # Store chat content
        output.append({"context_name": file_path, "question": question})
        print("\n\n****")
        print(f"Question: {question}")
        print("****")
    except Exception as exc:
        print(f"OpenAI error\n{str(exc)}")

    return output


if __name__ == "__main__":
    openai.api_key = sys.argv[1]
    q_per_doc = int(sys.argv[2])
    reg_folder = (
        sys.argv[3]
        if len(sys.argv) >= 4
        else "/Users/myong/Documents/workspace/reginald/data/handbook"
    )
    data_name = "reg"

    # Load existing answers
    if not os.path.exists("collected_questions"):
        os.makedirs("collected_questions")
    try:
        output = pickle.load(
            open(f"collected_questions/{data_name}_generated.pickle", "rb")
        )
    except:
        output = {}

    reg_files = [f for f in os.listdir(reg_folder) if isfile(join(reg_folder, f))]
    for f in reg_files:
        print(f"Now working on {f}")
        output[f] = process_file(os.path.join(reg_folder, f), q_per_doc)
        pickle.dump(
            output,
            open("collected_questions/{}_generated.pickle".format(data_name), "wb"),
        )

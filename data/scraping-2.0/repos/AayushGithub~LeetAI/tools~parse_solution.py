# given an input in the following format, open the corresponding file 
# python parse_solution.py <input_folder>
# input_folder should contain the following file:
# solution.py
import sys
import openai
import os
from dotenv import load_dotenv
import json

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def parse_solution(input_folder):
    # open the solution file
    solution_file = open(input_folder + "/solution.py", "r")
    # The file has the following format:
    # Link - <leetcode link>
    # Question ID - <question id>
    # Question Name - <question name>
    # solution

    link = solution_file.readline().strip().split(" - ")[1]
    question_id = solution_file.readline().strip().split(" - ")[1]
    question_name = solution_file.readline().strip().split(" - ")[1]
    solution = solution_file.read()

    solution_file.close()

    # return the parsed data
    return link, question_id, question_name, solution

def generate_explaination(link, question_id, question_name, solution):
    prompt = f"This is my solution to the Leetcode question:\nLink - {link}\nQuestion ID - {question_id}\nQuestion Name - {question_name}\n\nSolution:\n```{solution}```\nGive me an explanation (as a markdown file) as to why it works and if it is correct:\n"
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )
    explanation = response["choices"][0]["message"]["content"]
    return explanation

def save_to_markdown(explanation, input_folder):
    # save the explanation to a markdown file
    explanation_file = open(input_folder + "/explanation.md", "w")
    explanation_file.write(explanation)
    explanation_file.close()

def main():
    input_folder = sys.argv[1]
    link, question_id, question_name, solution = parse_solution(input_folder)
    explaination = generate_explaination(link, question_id, question_name, solution)
    save_to_markdown(explaination, input_folder)

if __name__ == "__main__":
    main()
import requests
from bs4 import BeautifulSoup
import re
from openai import OpenAI

def fetch_and_convert_problem(url):
    """Fetch content from a URL and convert to plain text."""
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        return f"Error: Unable to fetch the webpage. {e}"

    soup = BeautifulSoup(response.text, 'html.parser')
    problem_text_html = soup.find('article')

    if not problem_text_html:
        return "Error: Problem text not found."

    return ' '.join(problem_text_html.stripped_strings)

def get_solution_from_gpt(problem_text):
    """Get solution from GPT based on the problem text."""
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are designed to solve Advent of Code problems. Given a problem return a fully function python program that solves the problem. Make sure that the code is executable and do not include anything else that would cause a compilation error."},
            {"role": "user", "content": problem_text},
        ]
    )
    return response.choices[0].message.content

def extract_first_python_block(text):
    """Extract the first Python code block from text."""
    pattern = r"```python(.*?)```"
    match = re.search(pattern, text, re.DOTALL)

    return match.group(1).strip() if match else None

def main():
    url = 'https://adventofcode.com/2023/day/1'
    problem_text = fetch_and_convert_problem(url)
    if problem_text.startswith("Error"):
        print(problem_text)
        return

    solution_text = get_solution_from_gpt(problem_text)
    solution_program = extract_first_python_block(solution_text)

    if solution_program:
        exec(solution_program)
    else:
        print("No Python solution found.")

if __name__ == "__main__":
    main()

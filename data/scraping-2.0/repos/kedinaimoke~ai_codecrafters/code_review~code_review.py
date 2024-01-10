import argparse
import os
from dotenv import load_dotenv
import random
import string
import sys
import webbrowser
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

api_key = os.getenv("API_KEY")

client = OpenAI(
    api_key=api_key,
)

"""
This is the template prompt used to guide the AI in generating code reviews. 
It instructs the AI to provide a comprehensive code review, considering various aspects of the code such as clarity, 
correctness, test coverage, and adherence to coding standards.
"""

PROMPT_TEMPLATE = f"""Thorough code reviews should examine the change in detail and consider its integration within the codebase. 
                    Assess the clarity of the title, description, and rationale for the change. Evaluate the code's correctness, 
                    test coverage, and functionality modifications. Verify adherence to coding standards and best practices. 
                    Provide a comprehensive code review of the provided diffs and score the developer out of a 100% mark, 
                    suggesting improvements and refactorings based on SOLID principles when applicable. Also state topics based
                    on the improvements that the developer can read up on and refer to for better developer craftsmanship. Please refrain 
                    from further responses until the diffs are presented for review."""

def add_code_tags(text):
    """
    This function adds code tags (<b><code></code>) around inline code blocks in the given text. 
    It utilizes regular expressions to identify code blocks and wraps them in the specified tags.

    Args:
        text (str): The text containing inline code blocks.

    Returns:
        str: The modified text with code tags added around inline code blocks.
    """
    import re
    matches = re.finditer(r"`(.+?)`", text)

    updated_chunks = []
    last_end = 0
    for match in matches:
        updated_chunks.append(text[last_end : match.start()])
        updated_chunks.append("<b>`{}`<\b>".format(match.group(1)))
        last_end = match.end()
    updated_chunks.append(text[last_end:])
    return "".join(updated_chunks)

def generate_comment(diff, chatbot_context):
    """
    This function generates a code review comment and scoring using the OpenAI API. 
    It utilizes the GPT-3.5-turbo model to generate comments based on the provided diff and chatbot context. 
    It handles retries in case of API errors.

    Args:
        diff (str): The diff containing the code changes to be reviewed.
        chatbot_context (list): The chatbot context, a list of previous interactions with the AI.

    Returns:
        tuple: A tuple containing the generated comment and the updated chatbot context.
    """
    chatbot_context.append({
        "role": "user",
        "content": f"Make a code review of the changes made in this diff: {diff}",
    })

    retries = 3
    comment = ""

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                messages=[
                        {
                            "role": "user",
                            "content": f"Make a code review of the changes made in this diff and score the developer's code out of 100: {diff}",
                        },
                        {
                            "role": "assistant",
                            "content": comment,
                        }
                    ],
                    model="gpt-3.5-turbo",
                        )
            
            comment = response.choices[0].message.content

        except Exception as e:
            if attempt == retries - 1:
                print(f"attempt: {attempt}, retries: {retries}")
                raise e
            else:
                print("OpenAI error occurred. Retrying...")
                continue

    chatbot_context = [
        {"role": "user",
          "content": f"Make a code review of the changes made in this diff and score the developer's code out of 100 while stating areas and topics the developer can read on to improve their skills: {diff}"},
        {"role": "assistant", "content": comment},
    ]
    return comment, chatbot_context


def create_html_output(title: str, description: str, changes: list, prompt: str):
    """
    Generates an HTML output file containing code review comments for the provided diffs. 
    It utilizes a random filename, generates HTML content with code highlighting, writes the content to the file, 
    and opens the file in the default web browser.

    Args:
        title (str): The title of the diff being reviewed.
        description (str): The description of the diff being reviewed.
        changes (list): A list of dictionaries containing diff information.
        prompt (str): The prompt used to guide the AI in generating code reviews.

    Returns:
        str: The path to the generated HTML output file.
    """
    random_string = "".join(random.choices(string.ascii_letters, k=5))
    output_file_name = random_string + "-output.html"

    title_text = f"\nTitle: {title}" if title else ""
    description_text = f"\nDescription: {description}" if description else ""
    chatbot_context = [
        {"role": "user", "content": f"{prompt}{title_text}{description_text}"},
    ]

    html_output = "<html>\n<head>\n<style>\n"
    html_output += "body {\n    font-family: Roboto, Ubuntu, Cantarell, Helvetica Neue, sans-serif;\n    margin: 0;\n    padding: 0;\n}\n"
    html_output += "pre {\n    white-space: pre-wrap;\n    background-color: #f6f8fa;\n    border-radius: 3px;\n    font-size: 85%;\n    line-height: 1.45;\n    overflow: auto;\n    padding: 16px;\n}\n"
    html_output += "</style>\n"
    html_output += '<link rel="stylesheet"\n href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/default.min.css">\n <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js"></script>\n'
    html_output += "<script>hljs.highlightAll();</script>\n"
    html_output += "</head>\n<body>\n"
    html_output += "<div style='background-color: #333; color: #fff; padding: 20px;'>"
    html_output += "<h1 style='margin: 0;'>AI code review</h1>"
    html_output += f"<h3>Diff to review: {title}</h3>" if title else ""
    html_output += "</div>"

    with tqdm(total=len(changes), desc="Making code review", unit="diff") as pbar:
        for i, change in enumerate(changes):
            diff = change["diff"]
            comment, chatbot_context = generate_comment(diff, chatbot_context)
            pbar.update(1)
            html_output += f"<h3>Diff</h3>\n<pre><code>{diff}</code></pre>\n"
            html_output += f"<h3>Comment</h3>\n<pre>{add_code_tags(comment)}</pre>\n"
    html_output += "</body>\n</html>"

    with open(output_file_name, "w") as f:
        f.write(html_output)

    return output_file_name

def get_diff_changes_from_pipeline():
    """
    Retrieves diff changes from the standard input pipeline or user input. 
    It simulates a code change for review if no actual diff is available.

    Returns:
        list: A list of dictionaries containing diff information.
    """
    piped_input = sys.stdin.read()
    diffs = piped_input.split("diff --git")
    diff_list = [{"diff": diff} for diff in diffs if diff]
    return diff_list

def main():
    """
    The main entry point for the AI code review script. This function parses command-line arguments, 
    processes diffs from the standard input pipeline, and generates an HTML output file containing code review comments.

    Args:
        None
    """
    title, description, prompt = None, None, None
    changes = get_diff_changes_from_pipeline()
    parser = argparse.ArgumentParser(description="AI code review script")
    parser.add_argument("--title", type=str, help="Title of the diff")
    parser.add_argument("--description", type=str, help="Description of the diff")
    parser.add_argument("--prompt", type=str, help="Custom prompt for the AI")
    args = parser.parse_args()
    title = args.title if args.title else title
    description = args.description if args.description else description
    prompt = args.prompt if args.prompt else PROMPT_TEMPLATE
    output_file = create_html_output(title, description, changes, prompt)
    try:
        webbrowser.open(output_file)
    except Exception:
        print(f"Error running the web browser, you can try to open the output file: {output_file} manually")

if __name__ == "__main__":
    main()


# git diff main -- main/code_test_case/code_test_case.py | python code_review.py

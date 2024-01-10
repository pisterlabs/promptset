import argparse
import logging
import os
import sys
import openai
import xml.etree.ElementTree as ElementTree
from dotenv import load_dotenv
from util import generate_initial_prompt, generate_diff
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential
from openai import OpenAIError

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

load_dotenv(dotenv_path="../.env")
openai.api_key = os.getenv("API_KEY")


@retry(
    wait=wait_random_exponential(multiplier=1, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(OpenAIError) | retry_if_exception_type(ElementTree.ParseError)
)
def get_review_result(messages: list[dict], model: str) -> tuple[str, str, dict]:
    result = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.8
    )
    LOGGER.debug(f"Result Content: {result.choices[0].message.content}")
    LOGGER.debug(f"Usage: {result.usage}")
    result_content = result.choices[0].message.content

    try:
        root = ElementTree.fromstring(result_content)
    except ElementTree.ParseError:
        # attempt to find xml inside the text
        LOGGER.debug(f"Parsing the response failed. Attempting to find the xml inside the response:\n{result_content}")
        start_tag = '<root>'
        end_tag = '</root>'

        start_index = result_content.find(start_tag)
        end_index = result_content.find(end_tag) + len(end_tag)
        LOGGER.debug(f"Start Index: {start_index}, End Index: {end_index}")
        if start_index != -1:
            xml_content = result_content[start_index:end_index]
            LOGGER.debug(f"XML Content: \n{xml_content}")
            root = ElementTree.fromstring(xml_content)
        else:
            root = None
    if root is not None:
        code = root.find('code').text
        explanation = root.find('explanation').text
        return code, explanation, result.choices[0].message
    else:
        return "", result.choices[0].message.content, result.choices[0].message


def check_to_continue(file: str, model: str, messages: list[dict]) -> None:
    user_decision = input("Would you like continue reviewing the file?[Type 'Y' for yes or any other key to exit]:")
    if user_decision.upper() == 'Y':
        review_code(file, model, messages)
    else:
        print("Exiting the program. Good Bye!")
        sys.exit(0)


def review_code(file: str, model: str, messages: list[dict]) -> None:
    try:
        with open(file, "r") as f:
            file_contents = f.read()
    except FileNotFoundError:
        LOGGER.error(f"Unable to read the file {file}")
        sys.exit(1)

    LOGGER.info(f"Reviewing {file}")

    messages.append({'role': 'user', 'content': f'Suggest a single change for the code: {file_contents}'})
    # todo: count tokens and warn
    improved_code, explanation, assistant_message = get_review_result(messages, model)
    messages.append(assistant_message)

    if improved_code:
        print(generate_diff(file_contents, improved_code))
        print(f"\nAssistant: {explanation}\n\n")
    else:
        print(f"\nAssistant: {explanation}\n\n")
        sys.exit(0)

    user_decision = input("Would you like to apply this change?[Y/N]:")

    if user_decision.upper() == 'Y':
        try:
            with open(file, "w") as f:
                f.write(improved_code)
        except FileNotFoundError:
            LOGGER.error(f"Unable to write to the file: {file}")
        check_to_continue(file, model, messages)
    elif user_decision.upper() == 'N':
        check_to_continue(file, model, messages)
    else:
        print("Invalid input. Skipping applying the change...")
        check_to_continue(file, model, messages)


def main():
    parser = argparse.ArgumentParser(description="Code Reviewer OpenAI API")
    parser.add_argument("file", help="The target file to review")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="The model to use(default: gpt-3.5-turbo)")
    args = parser.parse_args()

    try:
        review_code(args.file, args.model, generate_initial_prompt())
    except KeyboardInterrupt:
        print("Good Bye!!")


if __name__ == "__main__":
    main()

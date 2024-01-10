import io
import json
import logging
import os
import random
import re
import sys
import time
import traceback
from typing import Tuple
import uuid

import openai
import requests
from bs4 import BeautifulSoup

# Initialize the OpenAI API
# APIKEY IS SET VIA ENVIRONEMNT
openai.api_key = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL_NAME = "gpt-3.5-turbo-instruct"

logger = logging.getLogger(__name__)


class Config:
    OPENAI_MODEL_NAME = "gp3-5-turbo-instruct"
    MAX_CODE_GENERATION_RETRY_COUNT = 10
    MAX_RELEVANT_SNIPPETS = 3
    MAX_VERIFIER_TOKENS = 1000
    CODE_FORMAT = """```python\n# imports\nimport bs4 \n\ndef scraper(url: str) -> str:\n  # scraper logic goes here\n  pass\n\nif __name__ == '__main__':\n  url = "<DUMMY URL, REPLACE WITH ACTUAL URL>"\n  scraper(url)\n"""
    PROMPT = "job listings on this page"


def scrape(url: str) -> str:
    """
    Scrapes a website and returns the HTML.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # raise exception for bad responses
        return response.text
    except Exception as e:
        logger.error(f"Error while scraping {url}: {str(e)}")
        return ""


def strip_scripts_and_styles(html_source: str) -> str:
    """
    Removes all script and style tags from the HTML source.
    """
    soup = BeautifulSoup(html_source, "html.parser")
    [s.decompose() for s in soup("script")]
    [s.decompose() for s in soup("style")]
    logger.info(f"Cleaned HTML Source:\n{str(soup)}\n")
    return str(soup)


def get_relevant_snippets(html_source: str, prompt: str, results, max_snippets: int = 3) -> list:
    """
    Generates up to 'max_snippets' relevant snippets from the HTML source,
    after removing <script> and <style> tags, and only considering 2000 character long snippets.
    """
    relevant_snippets = []
    attempts = 0

    # Clean the HTML source by stripping <script> and <style> tags
    clean_html_source = strip_scripts_and_styles(html_source)
    # log the clean html source
    # logger.info(f"Clean HTML Source:\n{clean_html_source}")

    while len(relevant_snippets) <= max_snippets and attempts < max_snippets * 5:
        logger.info(
            f"Attempts: {attempts}, Found snippets: {len(relevant_snippets)}")
        # If the remaining text is shorter than 2000 characters, break
        if len(clean_html_source) <= 2000:
            break

        # Randomly select a start index for the snippet
        start_index = random.randint(0, len(clean_html_source) - 2000)
        snippet = clean_html_source[start_index: start_index + 2000]
        results["snippets_tried"].append(snippet)

        logger.info(f"Snippet (Attempt {attempts + 1}):\n{snippet}")

        if is_relevant_snippet(snippet, prompt):
            relevant_snippets.append(snippet)

        attempts += 1
        if len(relevant_snippets) == max_snippets:
            break

    if not relevant_snippets:
        logger.error("No relevant HTML snippets found after maximum attempts.")

    return relevant_snippets


def is_relevant_snippet(snippet: str, prompt: str) -> bool:
    """
    Check if a HTML snippet is relevant to the scraping prompt.
    """
    relevance_prompt = (
        f"Determine if the following HTML snippet is relevant to the prompt: {prompt}\n "
        f'Reply "YES" or "NO" accordingly, and explain why.'
        f"\nHTML Snippet:\n{snippet}"
    )
    response = openai.Completion.create(
        model=OPENAI_MODEL_NAME, prompt=relevance_prompt, max_tokens=100
    )
    relevance_text = response.choices[0].text.strip()

    logger.info(f"Relevance check response:\n{relevance_text}")

    return "YES" in relevance_text


def generate_code(
    debugging_info: str, previous_error: str, previous_code: str, prompt: str, website: str, relevant_snippets
) -> str:
    if not relevant_snippets:
        logger.error("No relevant HTML snippets were provided.")
        return "No relevant HTML snippets were provided."

    # Randomly choose one of the relevant snippets
    relevant_snippet = random.choice(relevant_snippets)
    logger.info(f"Using Relevant Snippet:\n{str(relevant_snippet)}")

    # Proceed with code generation using the relevant snippet
    instruct_prompt = (
        f"Please provide Python code to scrape the website and PRINT OUT: {prompt} as JSON.\n"
        f"Based on the following relevant HTML snippet from somewhere in the webpage:\n```{relevant_snippet}```\n"
        f"The code should take in this link `{website}` and PRINT out the requested data.\n"
        f"Generate only the code, not any example usages or output, and you MUST wrap your answer in a markdown code block.\n"
        f"Generate it in this format.\n"
        f"```python\n# imports\nimport bs4 \n\ndef scraper(url: str) -> str:\n  # scraper logic goes here\n  pass\n\nif __name__ == '__main__':\n  url = \"<DUMMY URL, REPLACE WITH ACTUAL URL>\"\n  scraper(url)\n"
        f"START CONTEXT\n"
        f"This is the debugging info:\n```{debugging_info}```\n"
        f"This is your previous error:\n```{previous_error}```\n"
        f"This is your previous code:\n```python\n{previous_code}```\n"
        f"END CONTEXT\n"
    )
    logger.info(f"Generation Prompt:\n{instruct_prompt}")
    response = openai.Completion.create(
        model=OPENAI_MODEL_NAME, prompt=instruct_prompt, max_tokens=2000
    )

    # Log raw response
    logger.info(f"Raw response:\n{response}")

    # Extract content between backticks from the model's response
    content_match = re.search(
        r"```python\n(.*?)```", response.choices[0].text, re.DOTALL
    )

    if not content_match:
        logger.info("Code block not found. Trying without python syntax.")
        content_match = re.search(
            r"```python\n(.*?)$", response.choices[0].text, re.DOTALL
        )

    if not content_match:
        logger.info("Code block not found. Trying without python syntax.")
        content_match = re.search(
            r"```\n(.*?)```", response.choices[0].text, re.DOTALL)

    if not content_match:
        logger.info(
            "Code block not found. Trying without both ended backticks.")
        content_match = re.search(
            r"```\n(.*?)$", response.choices[0].text, re.DOTALL)

    if content_match:
        code = content_match.group(1).strip()
        # Log the extracted code
        logger.info(f"Extracted Code:\n{code}")

    else:
        logger.info("Code block not found. Giving raw response.")
        code = response.choices[0].text

        logger.info(f"Extracted Code:\n{code}")

    return code
    # else:
    #     # If no code block is found, log the full response for manual inspection
    #     logger.error(
    #         f"Code block not found in response:\n{response.choices[0].text}")
    #     return "Failed to extract code from the AI's response."


def verifier(output: str, prompt: str) -> Tuple[bool, str]:
    instruct_prompt = (
        f"Please verify if the following output snippet:\n```\n{output[:500]}\n```\n"
        f"accurately fulfills the requirements based on the prompt:\n```\n{prompt}\n```. "
        f"Flag out any irregularities!\n"
        f"A valid output should be roughly JSON (not including braces is okay), and MUST NOT be an empty list and should have the content described by the prompt!\n"
        f'Respond with a brief explanation of your assessment, and then write either "YES" or "NO" in a markdown code block.'
    )
    # log the prompt
    logger.info(f"Verifier prompt:\n{instruct_prompt}")
    response = openai.Completion.create(
        model=OPENAI_MODEL_NAME, prompt=instruct_prompt, max_tokens=1500
    )
    # log the raw response
    logger.info(f"Verifier raw response:\n{response}")
    answer_text = response.choices[0].text
    # log the answer text
    logger.info(f"Answer text:{answer_text}")

    if "YES" in answer_text:
        return True, answer_text
    elif "NO" in answer_text:
        return False, answer_text
    else:
        return False, "Invalid response."


def runner(code: str, url: str) -> Tuple[str, str]:
    """
    Runs the Python code with the given website URL and captures the printed output.
    Returns a tuple containing the output and the specific line of code that caused the error, if any.
    """
    # Create a StringIO object to capture stdout
    captured_output = io.StringIO()
    # Initialize the error_message variable to capture stderr
    error_message = ""

    # Split the code into lines for later reference
    code_lines = code.splitlines()

    # Save the current stdout
    current_stdout = sys.stdout
    sys.stdout = captured_output

   # Add a global dictionary to pass to exec
    try:
        global_dict = {
            '__builtins__': __builtins__,
            'requests': requests,
            'BeautifulSoup': BeautifulSoup,
            'json': json,
        }
        # Execute the code within the provided global scope
        exec(code, global_dict)

        # After executing the code, call the 'scraper' function with the provided URL
        if 'scraper' in global_dict:
            global_dict['scraper'](url)
        else:
            raise Exception(
                "The 'scraper' function is not defined in the provided code.")
    except Exception as e:
        # Get the last exception information
        exc_type, exc_value, exc_traceback = sys.exc_info()
        # Extract the line number from the traceback
        tb_info = traceback.extract_tb(exc_traceback)
        # Get the last traceback object in the list
        last_call = tb_info[-1]
        error_line_number = last_call.lineno
        # Retrieve the offending line of code using the line number
        # -1 because list indices start at 0
        # catch if it's out of range!
        if error_line_number > len(code_lines):
            error_line_number = len(code_lines)
        error_line_code = code_lines[error_line_number - 1]
        error_message = f"Error on line {error_line_number}: {error_line_code}\n{exc_type.__name__}: {exc_value}"
    finally:
        # Restore stdout to its original state
        sys.stdout = current_stdout

        # Get the contents of the StringIO buffer
        output = captured_output.getvalue()
        # Close the StringIO buffer
        captured_output.close()

    # Return the output and the error message
    logger.info(f"Runner :\n{output}")
    logger.info(f"Runner error message:\n{error_message}")
    # if output is empty
    if not output and not error_message:
        output = "No output returned!"
        error_message = "No output was printed!"

    # if ["[]", "{}", ""] in output:
    if "[]" in output or "{}" in output:
        output = "[] or \{\} was printed!"
        error_message = "[] or \{\} was received. There should be data. Check if you are scraping correctly."

    # log output and error message
    return output, error_message.strip()


def debugger(code: str, error: str, html_snippet: str) -> str:
    instruct_prompt = (
        f"Provide give one best guess for how to fix this error: {error}.\n"
        f"Given the code:\n```{code}```\n"
        f"Given a HTML snippet:\n```{html_snippet}```\n"
    )
    # log the prompt
    logger.info(f"Debugger prompt:\n{instruct_prompt}")
    response = openai.Completion.create(
        model=OPENAI_MODEL_NAME, prompt=instruct_prompt, max_tokens=2000
    )
    # log the respones
    logger.info(f"Debugger raw response:\n{response}")
    answer_text = response.choices[0].text
    return answer_text


def generate_scraper(
    prompt: str,
    website: str,
    output_dir: str,
    results: dict = None,
    retry: int = 3,
    verbose: bool = False,
    output: str = "json",
    api_key: str = None,
    log: str = None,
) -> (int, bool, str):
    """
    Generates a web scraper using OpenAI's models.

    Returns:
    - whether a scraper was successfully generated
    - number of tries required to successfuly generate a model
      (if failed, will always be equal to the `retry` parameter)
    """
    html_source = scrape(website)
    # log original html source
    logger.info(f"Original HTML Source:\n{html_source}")
    previous_code = "There is no previous code."
    previous_error = "There is no previous error."
    debugging_info = "There is no debugging info."
    relevant_snippets = get_relevant_snippets(html_source, prompt, results)
    results['relevant_snippets'] = relevant_snippets
    results['source'] = ""

    attempts_taken = 0

    if not relevant_snippets:
        logger.error("No relevant HTML snippets were found.")
        return (-1, False, previous_code)

    for i in range(retry):
        attempts_taken += 1
        code = generate_code(debugging_info, previous_error, previous_code, prompt,
                             website, relevant_snippets)
        previous_code = code
        logger.info(f"Generated code (Attempt {i + 1}):\n{code}")

        result, error = runner(code, website)
        # log results and errors
        logger.info(f"Result (Attempt {i + 1}):\n{result}")
        logger.info(f"Error (Attempt {i + 1}):\n{error}")
        # After each attempt to generate code
        results['generated_code_tries'].append({
            'attempt': i + 1,
            'code': code,
            'error': error,
            'debugging_info': ""
        })

        if error:
            previous_error = error
            # Passing the generated code and error to the debugger
            debugging_info = debugger(
                code, error, random.choice(relevant_snippets))
            logger.error(f"Attempt {i + 1} failed. Error: {error}")
            logger.error(
                f"Debugging info (Attempt {i + 1}):\n{debugging_info}")
            if verbose:
                print(
                    f"Attempt {i + 1} failed. Debugging info: {debugging_info}")

            results["generated_code_tries"][-1]["debugging_info"] = debugging_info

            # # Delay before the next retry
            # if i < retry - 1:
            #     delay = (i + 1) * 1  # increasing delay with each retry
            #     logger.info(f"Waiting for {delay} seconds before retrying...")
            #     time.sleep(delay)
            continue

        verified, verifier_message = verifier(result, prompt)
        if verified:
            logger.info("Successfully generated a valid scraper.")
            logger.info(f"Generated result (Attempt {i + 1}):\n{result}")

            return (attempts_taken, True, previous_code)
        else:
            logger.warning(
                f"Output didn't match the prompt. Verifier Message (Attempt {i + 1}): {verifier_message}"
            )
            debugging_info = debugger(
                code, verifier_message, random.choice(relevant_snippets))
            # debugging_info = (
            #     verifier_message
            # )
            # debugging_info = (
            #     f"Output didn't match the prompt. Expected: {prompt}. Got: {result}"
            # )
            logger.error(
                f"Debugging info (Attempt {i + 1}):\n{debugging_info}")
    logger.error("Failed to generate a valid scraper after max retries.")
    return (-1, False, previous_code)


def setup_logging(run_id, output_dir):
    # Create a 'runs' directory within the output directory
    runs_dir = os.path.join(output_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    # Create a run-specific folder within the 'runs' directory
    run_folder = os.path.join(runs_dir, str(run_id))
    os.makedirs(run_folder, exist_ok=True)

    # Create a unique filename for the log file inside the run-specific folder
    filename = os.path.join(run_folder, f"generation_log.log")

    # Configure the logging to use the new filename
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(filename, mode='a'),
            logging.StreamHandler()
        ]
    )


def load_dataset(file_path):
    """
    Loads the dataset from a JSON file.
    """
    with open(file_path, 'r') as file:
        return json.load(file)


def validate_scraper_on_websites(scraper_code, websites, prompt, results):
    """
    Validates the generated scraper on a list of websites.
    Saves the HTML source, the output of the scraper, and the GPT verifier's response.
    """
    validation_results = {}
    count = 0
    succ_count = 0
    for website in websites:
        count += 1
        logger.info(f"VALIDATING scraper on website: {website}")

        # Scrape the website and get the HTML source
        html_source = scrape(website)

        # Run the scraper and get the results
        result, error = runner(scraper_code, website)

        # If there's an error running the scraper, log it and continue to the next website
        if error:
            validation_results[website] = {
                'success': False,
                'error': error,
                'html_source': "redacted",
                'scraper_output': result,
                'verifier_response': 'N/A'  # If there's an error, the verifier isn't run
            }
            continue

        # Run the verifier
        verified, verifier_response = verifier(result, prompt)
        validation_results[website] = {
            'success': verified,
            'error': None if verified else 'Output did not match the prompt',
            'html_source': "redacted",
            'scraper_output': result,
            'verifier_response': verifier_response
        }
        if verified:
            succ_count += 1
        # log the validation results
        logger.info(f"Validation results:\n{validation_results[website]}")
    # results["test_results"] = validation_results
    results["test_count"] = count
    results["test_succ_count"] = succ_count
    return validation_results


def main():
    # Check if command line arguments were provided
    if len(sys.argv) < 3:
        print("Usage: python script.py <path_to_dataset_json> <dataset_name>")
        sys.exit(1)

    # Load the dataset
    path_to_dataset = sys.argv[1]
    dataset_name = sys.argv[2]

    # Load the dataset
    # specify the actual path to your dataset.json
    dataset = load_dataset(path_to_dataset)

    # Extract features and data from the dataset
    features = dataset['features']
    item_description = dataset['item_description']
    data_websites = dataset['data']

    prompt = (
        f"Extract {', '.join(features)} from the {item_description} this page"
    )

    # Use the first website for scraper generation
    website_to_generate = data_websites[0]
    output_dir = f"output/{dataset_name}"
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    run_id = uuid.uuid4()
    setup_logging(run_id, output_dir)

    results = {
        'dataset': dataset_name,
        'run': str(run_id),
        'source': "",
        'snippets_used': -1,
        'attempts': -1,
        'snippets_tried': [],
        'relevant_snippets': [],  # This will be populated during snippet generation
        'generated_code_tries': [],
        'final_scraper_code': None,
        'test_count': [],
        'test_succ_count': [],
        'test_results': {}
    }

    # Generate the scraper using the first website
    n_tries, success, generated_code = generate_scraper(
        prompt, website_to_generate, output_dir, results, verbose=True, retry=10)

    results['attempts'] = n_tries
    results['snippets_used'] = len(results['relevant_snippets'])

    # save snippets_tried, relevant_snippets, generated_code_tries, test_results into accessory json file
    with open(f"{output_dir}/runs/{run_id}/accessory.json", 'w', encoding='utf-8') as file:
        json.dump(results, file, indent=4)

    # delete snippets_tried, relevant_snippets, generated_code_tries, test_results from dictinoary
    del results['snippets_tried']
    del results['relevant_snippets']
    del results['generated_code_tries']
    del results['test_results']

    if success:
        logger.info(f'Successfully generated a scraper in {n_tries} tries.')
        results["final_scraper_code"] = generated_code
        # Validate the scraper on the rest of the websites

        os.makedirs(output_dir, exist_ok=True)
        # filename = os.path.join(output_dir, run_id, "scraper.py")
        with open(f"{output_dir}/runs/{run_id}/scraper.py", "w+") as f:
            f.write(generated_code)

        validation_results = validate_scraper_on_websites(
            generated_code, data_websites[1:], prompt, results)
        logger.info("Validation Results:", validation_results)

        os.makedirs(output_dir, exist_ok=True)
        # Save validation results
        with open(f"{output_dir}/runs/{run_id}/validation_results.json", 'w', encoding='utf-8') as file:
            json.dump(validation_results, file, indent=4)
    else:
        logger.error(f'Could not generate a scraper after {n_tries} tries.')

    # save the results
    with open(f"{output_dir}/runs/{run_id}/results.json", 'w', encoding='utf-8') as file:
        json.dump(results, file, indent=4)


if __name__ == "__main__":
    main()

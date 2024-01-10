import io
import re
import requests
import os
import sys
import openai
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from web_backend.db.jobs_util import JobStatus, update_job_status, get_job, update_job_response, update_job_selenium_code, update_selenium_output  # noqa: E402


def format_url(url: str) -> str:
    if url.startswith('http://') or url.startswith('https://'):
        # URL already starts with http:// or https://, so return it as is
        return url
    else:
        # URL doesn't start with http:// or https://, so prepend http:// and
        # return it
        return 'http://' + url


def extract_code_from_chat_gpt(prompt: str) -> str:
    pattern = r"```(?:python)?\n?(.+?)```"
    match = re.search(pattern, prompt, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None


def create_screenshot_directory(job_id: str) -> None:
    directory = f"src/web_backend/static/screenshots/{job_id}"
    os.makedirs(directory, exist_ok=True)
    return directory


def get_llm_response(job_id: str, url: str, instructions: str,
                     error_msg: str = None, old_code: str = None) -> str:
    # make dir if it doesn't exist already
    create_screenshot_directory(job_id)
    openai.organization = os.environ.get("OPENAI_ORG")
    openai.api_key =  os.environ.get("OPENAI_API_KEY")

    # TODO: (we will assume that we have one iteration of QA for now)
    properly_formed_url = format_url(url)
    response = requests.get(properly_formed_url)
    if response.status_code == 200:
        html = response.text
    else:
        # TODO: add error message to jobs table
        update_job_status(job_id, JobStatus.FAILED)
        return

    error_correction = "An error occured the first time you wrote this code: {}. Could you try rewriting the code to avoid the error? This was your original code: {}".format(
        error_msg, old_code) if error_msg is not None else ''

    prompt = 'Write headless selenium code in Python using Chrome to "{}" from the page "{}" for the following HTML code. After each action take a screenshot and save it to a directory called "src/web_backend/static/screenshots" (the screenshots should be named in increasing order ex: 1.png, 2.png, 3.png etc) and in a folder called "{}" (you will need to make the directory). Assume you do not need to specify a path to the webdriver. Please use the `find_element` function instead of functions that start with "find_element_by" as "find_element_by" has been removed. {} HTML Code: ```{}```'.format(
        instructions, url, str(job_id), error_correction, html)

    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[{"role": "assistant", "content": prompt}]
    )
    llm_response = chat.choices[0].message['content']
    return llm_response


def execute_selenium(job_id: str, url: str,
                     instructions: str, raw_code: str) -> str:
    # Redirect stdout to a buffer
    stdout_buffer = io.StringIO()
    sys.stdout = stdout_buffer

    # Execute the code
    try:
        exec(raw_code)
    except Exception as e:
        update_selenium_output(job_id, stdout_buffer.getvalue())
        sys.stdout = sys.__stdout__
        print(stdout_buffer.getvalue())
        # Redirect stdout to a buffer
        stdout_buffer = io.StringIO()
        sys.stdout = stdout_buffer
        # try one more time
        llm_response = get_llm_response(
            job_id, url, instructions, str(e), raw_code)
        update_job_response(job_id, llm_response)
        raw_code = extract_code_from_chat_gpt(llm_response)
        update_job_selenium_code(job_id, raw_code)
        exec(raw_code)

    # Restore stdout and get the result
    sys.stdout = sys.__stdout__
    result = stdout_buffer.getvalue()
    if len(result) == 0:
        result = 'Success!'
    return result


def process_request(job_id: str, url: str, instructions: str) -> None:
    job = get_job(job_id)
    if job['status'] == JobStatus.PENDING.value:
        update_job_status(job_id, JobStatus.PLANNING_QA)

        llm_response = get_llm_response(job_id, url, instructions)
        update_job_response(job_id, llm_response)
        raw_code = extract_code_from_chat_gpt(llm_response)
        update_job_selenium_code(job_id, raw_code)
        # once we get selenium response we can set to executing
        update_job_status(job_id, JobStatus.EXECUTING_QA)

    job = get_job(job_id)
    if job['status'] == JobStatus.EXECUTING_QA.value:
        result = execute_selenium(job_id, url, instructions, raw_code)
        update_selenium_output(job_id, result)
        update_job_status(job_id, JobStatus.COMPLETED)

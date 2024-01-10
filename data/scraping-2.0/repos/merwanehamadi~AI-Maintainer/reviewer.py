from __future__ import annotations

import os
import re
from textwrap import dedent

import requests
import json
import openai
from openai.error import APIError, RateLimitError, Timeout
import time
from dotenv import load_dotenv


def load_envs():
    """
    Loads env variables from .env file
    """

    load_dotenv()


def review_pr(pr_link: str) -> str:
    """
    A function that takes in code and suggestions and returns a response from create
      chat completion api call.

    to get diff add ".diff to the end of the PR" and then make an http request

    Make a review comment on a pull request

    Request change or approve PR with the github api

    Parameters:
        suggestions (list): A list of suggestions around what needs to be improved.
        code (str): Code to be improved.
    Returns:
        A result string from create chat completion. Improved code in response.
    """
    # use requests to get the pr diff
    diff_link = pr_link + '.diff'
    response = requests.get(diff_link)
    if response.status_code != 200:
        raise ValueError(f'Invalid response status: {response.status_code}. '
                         f'Response text is: {response.text} ')
    diff = response.text
    print(f"diff: {diff}")

    # now we need to make llm call to evaluate the reponse
    llm_response = _process_diff(diff)
    # llm_response = "acceptable stuff here"
    print(f"diff response: {llm_response}")
    _push_review(llm_response, pr_link)

    return "Successfully reviewed PR."


def _process_diff(diff):
    """
    Given a diff
    """
    system_prompt = dedent(f"""
      Instructions:

        You are a github project maintainer and pull request reviewer. Your job is to review pull requests and determine if they are acceptable or not. When diffs are not acceptable, you must provide feedback to the contributor on how to improve their diff.
        You are going to be provided with a pull request diff from a contributor to review. Your job is to determine if the diff is acceptable or not according to the project's "pull request guidelines" which will be provided below.
        You have very high standards for accepting a diff. The project's guidelines for acceptable PRs are as follows:

        ```
        Pull Request Guidelines:

        - Pull requests should be atomic and focus on a single change.
        - Pull requests should include tests. We automatically enforce this with [CodeCov](https://docs.codecov.com/docs/commit-status)
        - Classes and methods should have docstrings.
        - Pull requests should have a descriptive title and description. The description should explain what the pull request does.
        - Pull requests should not include any unrelated or "extra" small tweaks or changes.
        ```

        You receive a pull request from a contributor. The diff for the pull request is as follows:

        ```
        {diff}
        ```

        If the diff is acceptable, respond with "Acceptable". If the diff is not acceptable, respond with "Request Changes" and explain the needed changes.

    """)
    model = "gpt-4"
    # parse args to comma separated string
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": diff},
    ]

    response = create_chat_completion(model=model, messages=messages, temperature=0)
    return response


def _push_review(review, pr_link):
    """
    Push review to github
    link: https://api.github.com/repos/{{owner}}/{{repo}}/pulls/{{pull_number}}/reviews
    Body: {
        "event": "APPROVE",
        "body": "review"
    }
        Body: {
        "event": "REQUEST_CHANGES",
        "body": "review"
    }
    Post for both
    The response either starts with either "acceptable" or "request changes"
    If it doesn't we throw an error and let AutoGPT process it.
    We then get the response after that and then push it to github with the API requests shown above
    """
    accepted = False

    info = extract_github_info(pr_link)

    review = review.strip()
    if review.lower().startswith("acceptable"):
        accepted = True
        tail_of_review = review[len("acceptable"):]
    elif review.lower().startswith("request changes"):
        tail_of_review = review[len("request changes"):]
    else:
        raise ValueError(f"Invalid response: {review}. It must start with either 'acceptable' or 'request changes'")

    # now we need to push the review to github
    body = {
        "event": "APPROVE" if accepted else "REQUEST_CHANGES",
        "body": tail_of_review,
    }
    # print(f"Bearer {os.getenv('GITHUB_REVIEWER_TOKEN')}")
    print('Pushing review: ', body)
    print('url: ', f"https://api.github.com/repos/{info['owner']}/{info['repo']}/pulls/{info['pull_id']}/reviews")
    response = requests.post(
        f"https://api.github.com/repos/{info['owner']}/{info['repo']}/pulls/{info['pull_id']}/reviews",
        data=json.dumps(body),
        headers={
            "Authorization": f"Bearer {os.getenv('GITHUB_REVIEWER_TOKEN')}",
            "Cookie": f"logged_in=no",
            "Content-Type": "application/json",
            'X-GitHub-Api-Version': '2022-11-28',
            'Accept': 'application/vnd.github.html+json',
            'Accept-Encoding': 'gzip, deflate, br',
        }
    )
    if response.status_code != 200:
        raise ValueError(f'Invalid response status: {response.status_code}. '
                         f'Response text is: {response.text} ')


def extract_github_info(url):
    pattern = r'https://github.com/([^/]+)/([^/]+)/pull/(\d+)'
    match = re.match(pattern, url)

    if match:
        owner, repo, pull_id = match.groups()
        return {
            'owner': owner,
            'repo': repo,
            'pull_id': int(pull_id)
        }
    else:
        return None


def create_chat_completion(
    messages: List[Message],  # type: ignore
    model,
    temperature: float = None,
) -> str:
    """Create a chat completion using the OpenAI API

    Args:
        messages (List[Message]): The messages to send to the chat completion
        model (str, optional): The model to use. Defaults to None.
        temperature (float, optional): The temperature to use. Defaults to 0.9.
        max_tokens (int, optional): The max tokens to use. Defaults to None.

    Returns:
        str: The response from the chat completion
    """
    if temperature is None:
        temperature = 0

    num_retries = 5
    warned_user = False
    print(
        f"Creating chat completion with model {model}, temperature {temperature}"
    )
    response = None
    resp = None
    for attempt in range(num_retries):
        backoff = 2 ** (attempt + 2)
        try:
            print(f"Attempt {attempt + 1}/{num_retries}")
            print(f'model: {model}')
            print(f'messages: {messages}')
            print(f'temperature: {temperature}')
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        except RateLimitError:
            print(
                f"Error: ", f"Reached rate limit, passing..."
            )
            if not warned_user:
                print(
                    f"Please double check that you have setup a PAID OpenAI API Account. "
                    + f"You can read more here: https://docs.agpt.co/setup/#getting-an-api-key"
                )
                warned_user = True
        except (APIError, Timeout) as e:
            if e.http_status != 502:
                raise
            else:
                print(
                    f"Error: ",
                    f"API Bad gateway. Waiting {backoff} seconds...",
                )
                time.sleep(backoff)
            if attempt == num_retries - 1:
                raise
        if response is None:
            print(
                "FAILED TO GET RESPONSE FROM OPENAI",
                "Auto-GPT has failed to get a response from OpenAI's services. "
                + f"Try running Auto-GPT again, and if the problem the persists try running it with `--debug`.",
            )
        resp = response.choices[0].message["content"]
        break
    if resp is None:
        raise ValueError("Invalid response from OpenAI after 5 retries")
    return resp


if __name__ == "__main__":
    load_envs()
    review_pr("https://github.com/merwanehamadi/Auto-GPT/pull/301")

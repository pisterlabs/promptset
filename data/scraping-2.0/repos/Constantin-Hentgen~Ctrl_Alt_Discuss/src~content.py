from json import loads, dumps, dump
from os import makedirs, getenv
from datetime import datetime
from openai import OpenAI
import requests
import sys

# local imports
from configuration import (
    MISTRAL_LLM_ENDPOINT,
    MISTRAL_LLM_API_KEY,
    OPENAI_API_KEY_LLM,
    CUSTOM_INTRO_DATA,
    CUSTOM_OUTRO_DATA,
    GITHUB_REPO_OWNER,
    GITHUB_REPO,
    ARTICLE_URL,
    OUTPUT_PATH,
    OPENAI_LLM,
    LLM_CHOICE,
    SUBJECT,
)


from prompts.introduction import SYSTEM_PROMPT as introduction_system_prompt
from prompts.development import SYSTEM_PROMPT as development_system_prompt
from prompts.conclusion import SYSTEM_PROMPT as conclusion_system_prompt
from prompts.plan_attack import SYSTEM_PROMPT as plan_system_prompt
from prompts.metadata import (
    SYSTEM_PROMPT as metadata_system_prompt,
)

from rss import fetch_article_content


def content_validator(func):
    def wrapper(*args, **kwargs):
        while True:
            display_name = func.__name__.replace("generate_", "").replace("_", " ")
            print(f"\nGenerating {display_name}...")
            result = func(*args, **kwargs)
            preview = dumps(result, indent=2)
            print(f"\n{preview}\n")
            response = input(
                "Are you satisfied with the result? (yes/retry/stop): "
            ).lower()
            if response == "yes":
                break
            elif response == "stop":
                print("Stopping the program.")
                sys.exit(0)
        print()
        return result

    return wrapper


def get_latest_release_tag():
    endpoint = f"https://api.github.com/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO}/releases/latest"
    response = requests.get(endpoint)

    if response.status_code == 200:
        release_data = response.json()
        tag_name = release_data.get("tag_name")
        return tag_name
    else:
        print(f"Failed to retrieve latest release. Status code: {response.status_code}")
        return None


def generate_content(system_prompt: str, user_prompt: str) -> str:
    match LLM_CHOICE:
        case "openai":
            client = OpenAI(api_key=OPENAI_API_KEY_LLM)
            response = client.chat.completions.create(
                model=OPENAI_LLM,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {"role": "user", "content": user_prompt},
                ],
            )

            return loads(response.choices[0].message.content)
        case "mistral":
            endpoint = MISTRAL_LLM_ENDPOINT
            api_key = MISTRAL_LLM_API_KEY
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }

            data = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }

            response = requests.post(endpoint, headers=headers, json=data)
            output = response.json()["choices"][0]["message"]["content"]
            return loads(output)


@content_validator
def generate_plan() -> list:
    source = fetch_article_content(article_url=ARTICLE_URL)
    prompt = f"The target audience is ignorant in this subject so make the parts so focus on the popularization aspect."

    if source is not None:
        prompt += f"\n build the plan with this source : \n {source}"

    result = generate_content(system_prompt=plan_system_prompt, user_prompt=prompt)
    return result["plan"]


@content_validator
def generate_introduction(plan: str) -> list:
    prompt = (
        f"{plan}\n additional informations for the introduction : {CUSTOM_INTRO_DATA}"
    )
    return generate_content(
        system_prompt=introduction_system_prompt, user_prompt=prompt
    )["script"]


@content_validator
def generate_development(plan: list, introduction: list) -> list:
    development = []

    for part in plan:
        prompt = str(
            (
                f"Follow this description : \n"
                f"{part['description']}\n"
                f"Here is the previous parts of the development :\n"
                f"{development}"
            )
        )

        development.append(
            generate_content(
                system_prompt=development_system_prompt, user_prompt=prompt
            )["script"]
        )

    development = [item for part in development for item in part]
    return development


@content_validator
def generate_conclusion(introduction: str, development: list) -> list:
    prompt = f"introduction : {introduction}, development : {development}, custom information : {CUSTOM_OUTRO_DATA}"
    return generate_content(system_prompt=conclusion_system_prompt, user_prompt=prompt)[
        "script"
    ]


@content_validator
def generate_metadata() -> dict:
    metadata = generate_content(
        system_prompt=metadata_system_prompt, user_prompt=SUBJECT
    )

    now = datetime.now()
    formatted_time = now.strftime("%d/%m/%Y %H:%M:%S")
    version = get_latest_release_tag()

    output = {
        "datetime": formatted_time,
        "version": version,
        "llm": OPENAI_LLM,
        "title": metadata["title"],
        "description": metadata["description"],
        "article_url": ARTICLE_URL,
        "thumbnail_prompt": metadata["thumbnail_prompt"],
        "folder_name": metadata["folder_name"],
    }

    return output


def generate_podcast_content() -> dict:
    plan = generate_plan()
    introduction = generate_introduction(plan=plan)
    development = generate_development(introduction=introduction, plan=plan)
    conclusion = generate_conclusion(
        introduction=introduction,
        development=development,
    )
    metadata = generate_metadata()

    script = introduction + [{"name": "Transition"}] + development + conclusion

    podcast_path = f"{OUTPUT_PATH}\\{metadata['folder_name']}"
    makedirs(podcast_path)
    metadata_path = f"{podcast_path}\\metadata.json"

    with open(metadata_path, "w") as json_file:
        dump(metadata, json_file, indent=2)

    script_path = f"{podcast_path}\\script.json"

    with open(script_path, "w") as json_file:
        dump(script, json_file, indent=2)

    return {
        "thumbnail_prompt": metadata["thumbnail_prompt"],
        "folder_name": metadata["folder_name"],
    }

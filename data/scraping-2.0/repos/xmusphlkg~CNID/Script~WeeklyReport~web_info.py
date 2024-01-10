
from openai import OpenAI
from report_text import fetch_openai
import os
import time

def create_web_info(disease_name):
    info = openai_information(
        os.environ['WEBSITE_MAIN_CREATE'],
        os.environ['WEBSITE_MAIN_CHECK'],
        f"""Provide a comprehensive overview of the epidemiology of {disease_name}, 
        including its global prevalence, transmission routes, affected populations, and key statistics. 
        Include information on the historical context and discovery of {disease_name}.
        Highlight the major risk factors associated with {disease_name} transmission.
        Discuss the impact of {disease_name} on different regions and populations, 
        including variations in prevalence rates and affected demographics.""",
        disease_name,
        4096
    )
    # save to md file
    with open(f'../Report/information/{disease_name}.md', 'w') as f:
        f.write(info)

def openai_information(model_create, model_check, user_content, disease_name, token = 4096, max_retries=10, delay=1):
    """
    Fetches a response from the OpenAI API with automatic retries on failure.

    - model_create: The name of the model to use for the completion.
    - model_check: The name of the model to use for the check.
    - user_content: The content provided by the user for the completion.
    - token: The maximum number of tokens to generate.
    - max_retries: Maximum number of retries before giving up.
    - delay: Delay between retries in seconds.
    :return: The API response or None if all retries failed.
    """
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_API_BASE"],
    )

    attempt = 0
    messages_create = [{"role": "system",
                        "content": "You are an epidemiologist."},
                       {"role": "user",
                        "content": user_content}]
    while attempt < max_retries:
        content_raw = fetch_openai(model_create, client,
                                   messages_create,
                                   f"{disease_name} - information - Create",
                                   token, max_retries, delay)
        messages_check = [{"role": "system",
                            "content": "You are a language editing robot."},
                            {"role": "user",
                            "content": content_raw}]
        box_check = fetch_openai(model_check, client,
                                messages_check,
                                f"{disease_name} - information - Check",
                                token*2, max_retries, delay)
        # box_check = 'Yes'
        if "Yes" in box_check:
            return content_raw
        else:
            attempt += 1
            print(f"Retrying ({attempt}/{max_retries})...\n")
            print(f"box_check: {box_check}\n")
            print(f"box_content: {content_raw}\n")

    print(f"{disease_name} - information: retries reached. Failed to create response.")
    return None
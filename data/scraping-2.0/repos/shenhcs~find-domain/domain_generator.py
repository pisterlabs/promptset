import whois
import re
import threading
import os
import requests
import json
import openai

from dotenv import load_dotenv


model_lock = threading.Lock()

load_dotenv()
# llama_api_endpoint = os.getenv("LLAMA_API_ENDPOINT")
# llama_api_key = os.getenv("LLAMA_API_KEY")
openai_api_endpoint = "https://api.openai.com/v1/chat/completions"

openai_api_key = os.getenv("OPENAI_API_KEY")
if os.getenv("OPENAI_API_KEY") is None:
    raise EnvironmentError("OPENAI_API_KEY env not defined.")


# TODO: Should be a class...

BAD_DESCRIPTION = "Bad description"


def proc_txt(text):
    if BAD_DESCRIPTION in text:
        raise ValueError(BAD_DESCRIPTION)

    # extract domain names
    dnlist = [item.strip() for item in text.split(",")]
    # only alphanumeric or hyphen
    dnlist = [domain for domain in dnlist if re.match(
        r"^[a-zA-Z0-9-]+$", domain)]
    # capitalize the first letter and keep the rest as it is
    dnlist = [domain[0].upper() + domain[1:] for domain in dnlist]
    return dnlist


def get_inference(description, num_domains=40):
    print("Generating domain names...")
    user_prompt = f"```{description}```"

    sys_prompt_content = f"You are David Ogilvy of domain name generation.\
    You will be provided with a text with description of a business idea, activity delimited by triple backticks.\
    If the text contains a description, generate a list of {num_domains} creative, catchy and fun domain names.\
    If a domain name consists of multiple words, the first letter of each word shoud be uppercase.\
    Do your best to create domain names in the language of description.\
    Provide them in comma separated values without domain extension.\
    If the text does not contain a description, simply write \"{BAD_DESCRIPTION}\"\
    "

    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": sys_prompt_content},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.9
    }

    response = requests.post(openai_api_endpoint, headers=headers, json=data)
    response.raise_for_status()

    completion = response.json()
    print("completion:", completion)
    result = proc_txt(completion['choices'][0]['message']['content'])
    return result


def generate_domains_without_extension(description, num_domains=30, max_retry=3):
    num_good_domains = 10
    # Clean up the description
    description = re.sub(r"[^a-zA-Z0-9\s-]", "", description)

    retry_count = 0
    domains = []
    while retry_count < max_retry:
        generated_domains = get_inference(description, num_domains)
        if len(generated_domains) == 0:
            print("No domains generated.")
            break

        domains = list(set(domains) | set(generated_domains))

        if len(domains) >= num_good_domains:
            return domains

        retry_count += 1
        print(f"Retrying ({retry_count}/{max_retry})...")

    print(
        f"Unable to generate {num_good_domains} domains after {max_retry} retries.")
    return domains


def add_extensions(domains_without_extensions, extensions):
    domains_with_extensions = []
    for domain_without_extensions in domains_without_extensions:
        for extension in extensions:
            domain_with_extension = domain_without_extensions + extension
            domains_with_extensions.append(domain_with_extension)
    return domains_with_extensions


def check_domain_availability(domain_name):
    # print("check domain availibility")
    try:
        w = whois.whois(domain_name)
        return False
    except whois.parser.PywhoisError:
        return True


# def generate_available_domains(description, extensions, num_domains=1):
#     print("generate available domains")
#     generated_domains = set()
#     available_domains = []
#     while len(available_domains) < num_domains:
#         try:
#             domains = generate_domains_without_extension(
#                 description, generated_domains, num_domains -
#                 len(available_domains)
#             )
#         except ValueError as e:
#             print(f"Error: {e}")
#             return available_domains
#         for domain in domains:
#             if domain in generated_domains:
#                 continue
#             generated_domains.add(domain)
#             for extension in extensions:
#                 domain_with_extension = domain + extension
#                 if check_domain_availability(domain_with_extension):
#                     print(f"{domain_with_extension} is available")
#                     available_domains.append(domain_with_extension)
#                     if len(available_domains) == num_domains:
#                         break
#                 else:
#                     print(f"{domain_with_extension} is NOT available")
#             if len(available_domains) == num_domains:
#                 break
#     return available_domains

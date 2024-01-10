import json
import os
import openai
import re
import random
import time
from dotenv import load_dotenv
from typing import List, Dict, TypedDict


# ==================================================================================================
# Load Parameters
# ==================================================================================================

# Load .env file
load_dotenv()

# Get the API key
openai_api_key = os.getenv("OPENAI_API_KEY", "")

# Use the API key
openai.api_key = openai_api_key
openai.Model.list()

# load memory directory
memory_dir = os.getenv("MEMORY_DIRECTORY", "local")
workspace_path = "./"
# The workspace_path is the path to the workspace directory.
if memory_dir == "production":
    workspace_path = "/tmp"
elif memory_dir == "local":
    workspace_path = "./"


class Message(TypedDict):
    role: str
    content: str
    
    
# ==================================================================================================
# API Interaction
# ==================================================================================================

def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (openai.error.RateLimitError,
                     openai.error.Timeout,
                     openai.error.ServiceUnavailableError,
                     openai.error.APIError,
                     openai.error.InvalidRequestError,
                     openai.error.APIConnectionError),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                print("error: \n", e)
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                print(f"Wait for {round(delay, 2)} seconds.")
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                print("exception: \n", e)
                raise e

    return wrapper


@retry_with_exponential_backoff
def chat_with_gpt3(messages: str | List[Message], temp=1.0, p=1.0, freq=0.0, presence=0.0, model="gpt-3.5-turbo") -> str:

    language_selected = os.getenv("LANGUAGE")

    if isinstance(messages, str):
        response = openai.ChatCompletion.create(
            model=f"{model}",
            messages=[
                {"role": "system", "content": "You are an web designer with the objective to identify search engine optimized long-tail keywords and generate contents, with the goal of generating website contents and enhance website's visibility, driving organic traffic, and improving online business performance."},
                {"role": "system", "content": f"You will be writing in {language_selected} language"},
                {"role": "user", "content": messages}
            ],
            temperature=temp,
            # max_tokens=2500,
            top_p=p,
            frequency_penalty=freq,
            presence_penalty=presence,
        )
        # print (response)
        return response.choices[0].message['content']
    elif isinstance(messages, List):
        # print("messages: ", messages)
        response = openai.ChatCompletion.create(
            model=f"{model}",
            messages=messages,
            temperature=temp,
            # max_tokens=2500,
            top_p=p,
            frequency_penalty=freq,
            presence_penalty=presence,
        )
        # print (response)
        return response.choices[0].message['content']
    
    
# ##==================================================================================================
# JSON Functions
# ##==================================================================================================

def sanitize_location(location: str) -> str:
    """
     Sanitizes location to prevent XSS.
     
     @param location - The location to sanitize.
     
     @return The sanitized location as a string in the format " %20 " or " %2C "
    """
    url_safe_address = location.replace(" ", "%20")
    url_safe_address = url_safe_address.replace(",", "%2C")
    return url_safe_address


def processjson(jsonf: str) -> Dict:
    """
     Processes a JSON string and returns the result. If the JSON cannot be parsed an empty string is returned
     
     @param jsonf - the JSON string to process
     
     @return the JSON string or an empty string if there was
    """
    startindex = jsonf.find("{")
    endindex = jsonf.rfind("}")
    if startindex == -1 or endindex == -1:
        return {}
    else:
        try:
            return json.loads(jsonf[startindex:endindex+1])
        except ValueError as e:
            print("Json load error:\n", jsonf[startindex:endindex+1])
            print(e)
            return {}

# ##===================================================================================================
# Content Generation Methods
# ##===================================================================================================


def get_industry(topic: str) -> str:
    """
     Get industry for keywords.
     
     @param topic - keyword from user input
     
     @return identified industry for keyword
    """
    prompt = f"Generate an industry for these keywords, no explanation is needed: {topic}"
    industry = chat_with_gpt3(prompt, temp=0.2, p=0.1)
    print("Industry Found")
    return industry


def get_audience(topic: str) -> List[str]:
    """
     Get a list of audiences for a topic.
     
     @param topic - The topic for which to get the audience.
     
     @return A list of target audiences for the topic. It is empty if user cancels
    """
    audienceList = []
    prompt = f"Generate a list of target audience for these keywords, no explanation is needed: {topic}"
    audience = chat_with_gpt3("Target Search", prompt, temp=0.2, p=0.1)
    audiences = audience.split('\n')  # split the keywords into a list assuming they are comma-separated
    audiences = [target.replace('"', '') for target in audiences]
    audiences = [re.sub(r'^\d+\.\s*', '', target) for target in audiences]
    audienceList.extend(audiences)
    print("Target Audience Generated")
    return audienceList


def get_location(topic: str) -> str:
    """
     Generate location from user keyword.
     @param topic - topic of the address.
     @return a string of the form " street / city / postcode/ state / country
    """
    print("Identifying Location..")
    prompt = f"Generate an address (Building number, Street name, Postal Code, City/Town name, State, Country) in one line for this keywords, no explanation is needed: {topic}"
    location = chat_with_gpt3(prompt, temp=0.2, p=0.1)
    print("Location Found")
    return location


def generate_long_tail_keywords(topic: str) -> List[str]:
    """
     Generate 5 SEO optimised long tail keywords related to the topic.
     
     @param topic - topic to generate long tail keywords for
     
     @return list of keywords for the topic as a list of string
    """
    keyword_clusters = []
    prompt = f"Generate 5 SEO-optimized long-tail keywords related to the topic: {topic}."
    keywords_str = chat_with_gpt3(prompt, temp=0.2, p=0.1)
    keywords = keywords_str.split('\n')  # split the keywords into a list assuming they are comma-separated
    keywords = [keyword.replace('"', '') for keyword in keywords]
    keywords = [re.sub(r'^\d+\.\s*', '', keyword) for keyword in keywords]
    keyword_clusters.extend(keywords)
    print("Keywords Generated")
    return keyword_clusters


def generate_title(company_name: str,
                   keyword: str) -> str:
    """
    Generate and return title for a given companies headline.

    @param company_name - The name of the company
    @param keyword - The keyword for the title to be generated.

    @return The title as a string
    """
    prompt = f"Suggest 1 SEO optimized headline about '{keyword}' for the company {company_name}"
    title = chat_with_gpt3(prompt, temp=0.7, p=0.8)
    title = title.replace('"', '')
    print("Titles Generated")
    return title


def generate_meta_description(company_name: str,
                              topic: str,
                              keywords: str) -> str:
    """
    Generate a meta description for a website based on a topic and keywords.
    
    @param company_name - Company name to be used in the message
    @param topic - Topic for which we want to generate a meta description
    @param keywords - Keywords that will be used in the meta description
    
    @return Meta description as a string
    """
    print("Generating meta description...")
    prompt = f"""
    Generate a meta description for a website based on this topic: '{topic}'.
    Use these keywords in the meta description: {keywords}
    """
    meta_description = chat_with_gpt3(prompt, temp=0.7, p=0.8)
    return meta_description


def generate_footer(company_name: str,
                    topic: str,
                    industry: str,
                    keyword: str,
                    title: str,
                    location: str) -> dict:
    """
     Generate a footer. We need to generate an email to the Google Maps site and the map's url so it can be embedded in the template
     
     @param company_name - The company name of the user
     @param location - The location of the user in the google maps
     
     @return The JSON representation of the template's footer
    """
    print("Generating footer")
    start = random.choice(["+601", "+603"])
    rest = "".join(random.choice("0123456789") for _ in range(8))  # we generate 8 more digits since we already have 2
    number = start + rest
    email = "info@" + company_name.lower().replace(" ", "") + ".com"
    address = location.replace("1. ", "", 1)
    url_location = sanitize_location(address)
    mapurl = f"https://maps.google.com/maps?q={url_location}&t=&z=10&ie=UTF8&iwloc=&output=embed"
    
    footer_json = {
        "map": {
            "map_src": ""
        },
        "footer": {
            "info": []
        }
    }
    footer_json['map']['map_src'] = mapurl
    footer_json['footer']['info'].extend([number, email, address])
    return footer_json


def generate_content(company_name: str,
                     topic: str,
                     industry: str,
                     keyword: str,
                     title: str,
                     location: str) -> str:
    """
    Generates content for the template. This is a function that takes care of the creation of the content
    
    @param company_name - The name of the company
    @param topic - The keyword of the users
    @param industry - The industry of the topic
    @param keyword - The keyword found
    @param title - The title of the content
    
    @return The JSON string of the content
    """

    print("Generating Content...")
    directory_path = os.path.join(workspace_path, "content")
    os.makedirs(directory_path, exist_ok=True)
    json1 = """
    {
        "banner": {
                "h1": "...",
                "h2": "...",
                "button": [
                    {
                        "name": "...", 
                        "layout": 1
                        "style": []
                    },
                    {
                        "name": "...",
                        "layout": 2
                        "style": []
                    }...
                ] (Pick from these: Learn More, Contact Us, Get Started, Sign Up, Subscribe, Shop Now, Book Now, Get Offer, Get Quote, Get Pricing, Get Estimate, Browse Now, Try It Free, Join Now, Download Now, Get Demo, Request Demo, Request Quote, Request Appointment, Request Information, Start Free Trial, Sign Up For Free, Sign Up For Trial, Sign Up For Demo, Sign Up For Consultation, Sign Up For Quote, Sign Up For Appointment, Sign Up For Information, Sign Up For Trial, Sign Up For Demo, Sign Up For Consultation, Sign Up For Quote, Sign Up For Appointment, Sign Up For Information, Sign Up For Trial, Sign Up For Demo, Sign Up For Consultation, Sign Up For Quote, Sign Up For Appointment, Sign Up For Information, Sign Up For Trial, Sign Up For Demo, Sign Up For Consultation,  Sign Up For Quote, Sign Up For Appointment, Sign Up For Information)
        },
        "about": {
                "h2": "About Us",
                "p": "..."
        },
        "blogs": {
            "h2": "... (e.g.: Our Services, Customer Reviews, Insights, Resources)",
            "post": [{
                    "h3": "...",
                    "p": "...",
                },
                {
                    "h3": "...",
                    "p": "...",
                },
                {
                    "h3": "...",
                    "p": "...",
                }
            ]
        },
        "faq":{
            "h2": "Frequently Asked Questions",
            "question": [{
                    "id": 1,
                    "h3": "...",
                    "p": "...",
                },
                {
                    "id": 2,
                    "h3": "...",
                    "p": "...",
                },
                {
                    "id": 3,
                    "h3": "...",
                    "p": "...",
                },
                {
                    "id": 4,
                    "h3": "...",
                    "p": "...",
                },
                {
                    "id": 5,
                    "h3": "...",
                    "p": "...",
                },...
            ]
        },
        "blog2": {
                "h2": "Our Mission",
                "p": "..."
        }
    }
    """
    prompt = f"""
    Create a SEO optimized website content with the following specifications:
    Company Name: {company_name}
    Title: {title}
    Industry: {industry}
    Core Keywords: {topic}
    Keywords: {keyword}
    Format: {json1}
    Requirements:
    1) Make sure the content length is 700 words.
    2) The content should be engaging and unique.
    3) The FAQ section should follow the SERP and rich result guidelines
    """
    content = chat_with_gpt3(prompt, temp=0.7, p=0.8, model="gpt-3.5-turbo-16k")
    return content


def content_generation(company_name: str,
                       topic: str,
                       industry: str,
                       keyword: str,
                       title: str,
                       location: str) -> dict:
    """
    Generates and returns content. This is the main function of the content generation process
    
    @param company_name - The name of the company
    @param topic - The topic of the industry to generate
    @param industry - The industry of the industry to generate
    @param keyword - The keyword of the industry to generate
    @param title - The title of the industry to generate
    @param location - The location of the industry to generate
    
    @return dict with meta information about the content
    """
    print("Starting Content Process")
    try:
        description = generate_meta_description(company_name, topic, keyword)
        content = generate_content(company_name, topic, industry, keyword, title, location)
        footer = generate_footer(company_name, topic, industry, keyword, title, location)
    except Exception as e:
        return {'error': str(e)}
    contentjson = processjson(content)
    updated_json = {"meta": {"title": title, "description": description}}
    updated_json.update(contentjson)
    updated_json.update(footer)
    print("Content Generated")
    # print(json.dumps(updated_json, indent=4))
    return updated_json

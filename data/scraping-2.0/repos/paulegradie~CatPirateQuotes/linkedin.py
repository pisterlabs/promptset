import requests
from dotenv import dotenv_values
import os
import openai
import json
from typing import Optional
import argparse
import random

def generate_quote(prompt: str, model: str):
    completion = openai.Completion.create(
        model=model,
        max_tokens=500,
        temperature=0.7,
        presence_penalty=0.6,
        prompt=prompt)
    inspirational_quote = completion.choices[0].text
    return inspirational_quote


def clean_text(text: str):
    return text.strip().lstrip("\"").rstrip("\"")


def post_to_linkedin(post_text: str, user_id: str, bearer_token: str, img: Optional[str] = None):
    linked_headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Content-Type": "application/json",
        "x-li-format": "json"
    }
    data = json.dumps({
        "author": f"urn:li:{user_id}",
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {
                    "text": post_text
                },
                "shareMediaCategory": "NONE"
            }
        },
        "visibility": {
            "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
        }
    })

    res = requests.request(
        "POST",
        "https://api.linkedin.com/v2/ugcPosts",
        headers=linked_headers,
        data=data)
    print(res)


def main(
        prompt: str,
        model: str,
        send_to_linkedin: bool,
        user_id: Optional[str] = None,
        bearer_token: Optional[str] = None):

    inspirational_quote = clean_text(generate_quote(prompt, model))
    print(inspirational_quote)

    if (send_to_linkedin and user_id and bearer_token):
        post_to_linkedin(inspirational_quote, user_id=user_id,
                         bearer_token=bearer_token)

def decorate_prompt(prompt: str, formatter: str):
    return formatter + prompt.lower() + " in around 100 words that will well received."

if __name__ == "__main__":
    CURIOSITY_PROMPTS = [
         "important concept in software engineering that should get more attention"
    ]
    INSPIRATIONAL_PROMPTS = [
        "improving culture in software engineering companies"
        "Overcoming obstacles and adversity",
        "Embracing change and growth",
        "The importance of self-care and self-love",
        "Turning failures into opportunities",
        "Building resilience and inner strength",
        "Finding inner peace and tranquility",
        "The value of perseverance and hard work",
        "The beauty of diversity and inclusivity",
        "The impact of kindness and compassion",
        "Letting go of fear and embracing courage",
        "Empowering others to succeed",
        "Trusting the journey and having faith",
        "Finding inspiration in nature and the world around us",
        "Cultivating healthy relationships and connections",
        "Learning and growing as a software engineering leader",
        "Learning and growing as a software engineer",
        "People struggling to find a job"]

    FUN_FACT_TOPICS = [
        "Space",
        "History",
        "Music",
        "Science",
        "Movies",
        "Geography",
        "Art",
        "Inventions",
        "Fashion",
        "Literature",
        "Mythology",
        "Technology",
        "Holidays",
        "Transportation",
    ]

    PROMPT_SETS = {
        0: {
            'formatter': "Write an uplifting and inspirational quote about ",
            'prompt_set': INSPIRATIONAL_PROMPTS,
            'suffix': ""
        },
        1: {
            'formatter': "Write a fun fact about ",
            'prompt_set': FUN_FACT_TOPICS,
            'suffix': "Start the prompt with the words: \"Fun Fact: \""
        }
    }

    num_prompts = len(PROMPT_SETS.keys())
    INDEX_CHOICE = random.randint(0, num_prompts - 1)

    prompt = PROMPT_SETS[INDEX_CHOICE]['prompt_set'][random.randint(0, len(PROMPT_SETS[INDEX_CHOICE]) - 1)]
    formatter = PROMPT_SETS[INDEX_CHOICE]['formatter']
    DEFAULT_PROMPT = decorate_prompt(prompt, formatter);
    print()
    print(DEFAULT_PROMPT)
    print()
    parser = argparse.ArgumentParser(
        prog='Cat Pirate Quotes - LinkedIn',
        description='Uses chatGPT to generate text and post it to LinkedIn',
        epilog='Use with caution - who knows what chatGPT will say!')
    parser.add_argument('-s', '--post', required=False, action='store_true')
    parser.add_argument('-m', '--model', required=False, default="text-davinci-003")
    parser.add_argument('-p', '--prompt', default=DEFAULT_PROMPT)

    if os.path.exists(".env.development"):
        config = dotenv_values(".env.development")
        linkedin_user_id = config["LINKEDIN_ID"]
        linkedin_bearer_token = config["LINKEDIN_TOKEN"]
        openAiApiKey = config["OPENAI_APIKEY"]
        args = parser.parse_args()
    else:
        parser.add_argument('-u', '--linkedin-id', required=True)
        parser.add_argument('-l', '--linkedin-token', required=True)
        parser.add_argument('-o', '--openai-apikey', required=True)
        args = parser.parse_args()
        linkedin_user_id = args.linkedin_id
        linkedin_bearer_token = args.linkedin_token
        openAiApiKey = args.openai_apikey

    openai.api_key = openAiApiKey
    main(
        args.prompt,
        args.model,
        args.post,
        linkedin_user_id,
        linkedin_bearer_token)

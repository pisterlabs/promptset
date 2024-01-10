import openai
from dotenv import load_dotenv
import os
import random
import re


def init_openai():
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_SECRET')


def react_to_news():
    reaction_types = ["positively", "wholesomely", "angrily", "joyously"]
    react_instruction = f'React to the following piece of news: "President-elect Joe Biden introduced his slate of scientific advisers with the promise that they would summon “science and ' \
                        "truth” to combat the pandemic, climate crisis and other challenges.\n"
    init_openai()
    response = openai.Completion.create(engine="davinci-instruct-beta", prompt=react_instruction, max_tokens=50, stop="\n\n")
    return response["choices"][0]["text"]


def reply_to_thread(context):
    text_set = set()
    moods = ["happy", "supportive", "wholesome", "jokey", "comforting", "optimistic", "jovial", "kind", "sarcastic", "mocking"]
    thread = [context["root"]] + context["replies"]
    prompt_text = f'The following is a Twitter exchange between many people. All of the people are {random.choice(moods)}.\n\n'
    for tweet in thread:
        prompt_text += f'{tweet["author_handle"]}: {tweet["text"]}\n'
        text_set.add(tweet["text"][:50])
    prompt_text += f'{context["me_handle"]}:'

    init_openai()
    for i in range(5):
        response = openai.Completion.create(engine="davinci", prompt=prompt_text, max_tokens=50, temperature=0.9, stop="\n")["choices"][0]["text"].strip()
        if response[:50] not in text_set:
            return response
    return openai.Completion.create(engine="davinci", prompt="Tweet:", max_tokens=50, temperature=0.9, stop="\n")["choices"][0]["text"].strip()
#
#
# def random_new_tweet(_):
#     tweet_types = ["hot-takes", "jokes"]
#     selected_type_file = f'prompts/{random.choice(tweet_types)}.txt'
#     tweets_of_category = open(selected_type_file).read().splitlines()
#
#     # topics = ["Trump", "Facebook", "personality tests", "LeBron James", "smoking", "cooking", "astrology"]
#     prompt_tweets = random.sample(tweets_of_category, 5)
#     prompt = '\n'.join(prompt_tweets) + '\n'
#
#     init_openai()
#     response = openai.Completion.create(engine="davinci", prompt=prompt, max_tokens=50, stop="\n")["choices"][0]["text"]
#
#     category_filter_regex = "^.+?:(.*)"
#     regex_result = re.findall(category_filter_regex, response)
#     if regex_result and len(regex_result) >= 1:
#         response = regex_result[0].strip()
#     return response


def random_new_tweet(context):
    prompt = ""
    for tweet in context["tweets"]:
        cleaned = re.sub(r"#\w", "", tweet["text"])
        cleaned = cleaned.replace("labbur", "")
        prompt += f'{tweet["author_handle"]}: {cleaned}\n'
    prompt += f'{context["me_handle"]}:'
    init_openai()
    response = openai.Completion.create(engine="davinci", prompt=prompt, max_tokens=50, stop="\n")["choices"][0]["text"].strip()
    return response

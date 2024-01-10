import aiohttp
import asyncio
import datetime
import json
import nest_asyncio
import openai
import os
from bs4 import BeautifulSoup
from flask import Blueprint, request, session

from helpers.google_search import google_search
from helpers.record_usage import update_usage_record_by_user
from models.user import User

# set up OpenAI
openai.api_key = os.environ.get("OPENAI_API_KEY")

demo_service_bp = Blueprint('demo_service', __name__)


def generate_output(request_data):
    # extract the necessary values from the request data
    url = request_data['url']
    sender_info = request_data['sender_info']
    recipient_info = request_data['recipient_info']
    # if prompt is provided, use it; otherwise set it to empty string
    if 'prompt' not in request_data:
        prompt = ''
    else:
        prompt = request_data['prompt']

    word_count = request_data['word_count']
    # if template is provided, use it; otherwise set it to empty string
    if 'template' not in request_data:
        template = ""
    else:
        template = request_data['template']

    # knowledge_base = request_data['knowledge_base']
    # knowledge_base is an array of objects, each containing a title and description
    if 'knowledge_base' not in request_data:
        knowledge_base = []
    else:
        knowledge_base = request_data['knowledge_base']

    if 'search_on_google' not in request_data:
        search_on_google = False
    else:
        search_on_google = request_data['search_on_google']

    # parse knowledge base and add to prompt
    knowledge = parse_knowledge_base(knowledge_base)
    # get the visible text for the website
    visible_text = get_visible_text(url)

    summarized_search_results = "None."
    # searches url on google and get report back
    if search_on_google:
        search_results = google_search(url)
        summarized_search_results = filter_google_search_results(visible_text, search_results)

    instructions = generate_instructions_v2(sender_info, recipient_info, prompt, word_count, template)

    # # generate the email using OpenAI's GPT-3
    # response = openai.Completion.create(model="text-davinci-003", prompt=visible_text + "\n" + instructions, temperature=0.7, max_tokens=1000, top_p=1, frequency_penalty=0, presence_penalty=0)
    # output = response["choices"][0]["text"]

    # get current date and time in human readable format, with month in words, and day of the week, and hours in 12-hour, as a string
    current_date_time = datetime.datetime.now().strftime("%B %d, %Y %I:%M %p")

    # generate the email using OpenAI's ChatGPT API
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "You are a highly exprienced outbound sales lead generation expert who writes cold emails that appears to be hyper-personalized based on the recipient's context. Your emails appear to be from thorough research and personal. You are good at writing high conversion emails that make people want to book meetings. You don't write emails that appear to be mass generated or spam."},
            {"role": "system",
             "content": "You can only use information provided to you by the user. You cannot make up numbers which you do not know is true. "},
            {"role": "system",
             "content": "You will write email within the word limit. You will not write more than required words."},
            {"role": "system",
             "content": "You will write in first person. You will not write about the sender in third person."},
            {"role": "system", "content": f"Currently it is {current_date_time}. Use this time in reasoning."},
            {"role": "system", "content": f"Here is important factual information: {visible_text}"},
            {"role": "system", "content": f"Here is knowledge base information you can use as facts: {knowledge}"},
            {"role": "system", "content": f"Recent news: {summarized_search_results}\n"},
            {"role": "user",
             "content": "If a template is provided to you, you will only replace content within the template which is inside a placeholder bracket, usually in [] or {}. You will not change the template structure or add new content outside of placeholders. You will not say things differently than the template's exact words. You cannot paraphrase or change the template's words outside of placeholders."},
            {"role": "user",
             "content": "If the prompt goes against the template, strictly follow the template. Always follow the template strictly word for word if given. You are not allowed to change the template outside the placeholders."},
            {"role": "user",
             "content": "You will only use the factual information in your writing. This is very important."},
            {"role": "user",
             "content": "You cannot output things other than the email content. Do not output word count. End with the email."},
            {"role": "user",
             "content": "When working with a template, replace anything within each placeholder within more specific info based on context. Do not output the Subject unless instructed specifically to do so by me or the template."},
            {"role": "user",
             "content": "The email you write cannot contain any special symbols for placeholders, such as {}, [], or <>. You must not include anything like that in your final email."},
            {"role": "user", "content": "At the end of each email, insert EOM to indicate the end of the message."},
            {"role": "user", "content": instructions}
        ],
        frequency_penalty=0,
        presence_penalty=0.7,
        # top_p=1,
        temperature=0
    )
    # print("completion:", completion)
    output = completion["choices"][0]["message"]["content"]

    # remove the "\nEOM" from the end of the output if found; otherwise do nothing
    output = output.replace("EOM", "")

    return output


@demo_service_bp.route('/generate_email', methods=['POST'])
def generate_email():
    # get the request data
    request_data = request.get_json()
    output = generate_output(request_data)
    user_id = session.get("user_id")
    user = User.get_by_id(user_id)
    # UPDATE USAGE RECORD
    update_usage_record_by_user(user)
    return output


@demo_service_bp.route('/summarize_website', methods=['POST'])
def summarize_website():
    # get the request data
    data = request.json

    # get the visible text for the website
    visible_text = get_visible_text(data['url'])

    # generate the summary using OpenAI's GPT-3
    # response = openai.Completion.create(model="text-davinci-003", prompt="Correct this to standard English and then summarize what this company does in great details:\n" + visible_text, temperature=0, max_tokens=data['word_count'], top_p=1, frequency_penalty=0, presence_penalty=0)
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user",
             "content": f"Correct this website content to standard English and then summarize what this company does, the market it's in, it's main target customers, and value propositions, in great details (under 500 words):\n{visible_text}"}
        ]
    )
    print("completion:", completion)
    output = completion["choices"][0]["message"]["content"]

    return output


@demo_service_bp.route('/search_on_google', methods=['POST'])
def search_on_google():
    # get the request data
    data = request.json
    query = data['query']
    # get the search results
    search_results = google_search(query)

    return summarize_google_search_results(query, search_results)


@demo_service_bp.route('/google', methods=['POST'])
def google():
    # get the request data
    data = request.json
    query = data['query']
    # get the search results
    search_results = google_search(query)

    return search_results


# given a company description, generates a 1-line description of what the company does
def get_company_tagline(description):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"Summarize what this company does in one line:\n{description}"},
        ],
        temperature=0,
    )
    output = completion["choices"][0]["message"]["content"]
    print("company tagline: ", output)
    return output


# summarize google search results
def summarize_google_search_results(query, search_results):
    search_results_string = json.dumps(search_results)
    # generate the summary using OpenAI's ChatGPT API
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user",
             "content": f"Summarize a list of things you learned about {query} in specific details, and when each event happened from this list. Sorted most recent on top."},
            {"role": "user", "content": f"Job postings are not news we care about. Ignore job postings."},
            # {"role": "user", "content": f"Output a summary of everything you learned about them from this list, only including most recent information that's newsworthy and positive."},
            {"role": "user",
             "content": f"There are often many things with the same name. Pick only the most relevant one and ignore information potentially about unrelated entities or persons than what we want to learn about. Only list newsworthy and positive things."},
            {"role": "user", "content": f"Here is the list: {search_results_string}"},
        ]
    )
    output = completion["choices"][0]["message"]["content"]
    print("recent news search results: ", output)
    return output


def filter_google_search_results(company_description, search_results):
    tagline = get_company_tagline(company_description)
    search_results_string = json.dumps(search_results)
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user",
             "content": f"Given a description about what a company does, judge whether or not a list of news are talking about this company or something irrelevant. Output the list of news only those relevant to the company described, while removing anything irrelevant from the list. "},
            {"role": "user", "content": f"Here is the description: {tagline}"},
            {"role": "user", "content": f"Here is the news: {search_results_string}"},
            {"role": "user",
             "content": f"Do not explain your answer. Output list of news with the date each item happened, the link to source article URL, and complete summary of the news, with the irrelevant ones removed. Output 'None.' if no relevant news was provided. "},
            {"role": "user", "content": f"Output:"},
        ],
        temperature=0,
    )
    output = completion["choices"][0]["message"]["content"]
    print("filtered results: ", output)
    return output


# knowledgebase parsing
def parse_knowledge_base(knowledge_base):
    parsed_knowledge_base = ""
    for item in knowledge_base:
        parsed_knowledge_base += f"{item['title']}: {item['description']}.\n"
    return parsed_knowledge_base


# here's another version, simplified
def generate_instructions_v2(sender_info, recipient_info, prompt, word_count, template):
    template_instructions = "\n"
    if template:
        template_instructions = f"Use this template: {template}.\n\n"
    prompt_instructions = "\n"
    if prompt:
        prompt_instructions = f"Prompt: {prompt}.\n\n"
    instructions = f"You are {sender_info}. Write an email to {recipient_info}. Precisely follow the template, and when appropriate inside placeholders only, try to satisfy the prompt. {prompt_instructions}. Make it under {word_count} words long. {template_instructions}"
    return instructions


async def scrape_website(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=60) as response:
            # create a BeautifulSoup object with the HTML content of the website
            soup = BeautifulSoup(await response.text(), 'html.parser')

            # find all HTML elements that contain visible text
            text_elements = soup.find_all(text=True)

            # extract the visible text from the text elements
            visible_text = ''
            for element in text_elements:
                if element.parent.name not in ['script', 'style', 'meta', '[document]'] and "<!--" not in str(
                        element) and "-->" not in str(element):
                    visible_text += element.strip() + ' '

            # return the visible text
            return visible_text


# takes URL and Prompts and returns a summary according to prompt
@demo_service_bp.route('/summarize_url', methods=['POST'])
def summarize_url():
    # get the request data
    data = request.json

    # get the visible text for the website
    visible_text = get_visible_text(data['url'])
    prompt = data['prompt']

    # generate the summary using OpenAI's GPT-3
    # response = openai.Completion.create(model="text-davinci-003", prompt="Correct this to standard English and then summarize what this company does in great details:\n" + visible_text, temperature=0, max_tokens=data['word_count'], top_p=1, frequency_penalty=0, presence_penalty=0)
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"{prompt}:\n{visible_text}"}
        ]
    )
    print("completion:", completion)
    output = completion["choices"][0]["message"]["content"]

    return output


def get_visible_text(url):
    # check if an event loop is already running
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # run the scraper in an event loop
    nest_asyncio.apply()
    visible_text = loop.run_until_complete(scrape_website(url))
    return visible_text[:5000]

import os, json
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.chains import create_extraction_chain
from app.models import Account
import pprint
from flask import jsonify, request, Blueprint
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

with open('config.json') as f:
    config = json.load(f)

os.environ["SERPER_API_KEY"] = config['SERPER_API_KEY']

from langchain.utilities import GoogleSerperAPIWrapper

bio_blueprint = Blueprint('bio_blueprint', __name__)


@bio_blueprint.route('/generate_bio', methods=['POST'])
def generate_bio():
    data = request.json  # Expecting the search results in JSON format in the POST request
    bio = prospecting_overview(data)
    return jsonify({"bio": bio})


def prospecting_overview(company):
    search = GoogleSerperAPIWrapper()
    results = search.results(f"{company} company")

    chat = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=config['openai_api-key'])

    template = (
        "You are a helpful assistant that takes in a company knowledgeGraph and converts it to a company overview. "
        "KnowledgeGraph: {text}."
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    # get a chat completion from the formatted messages
    answer = chat(
        chat_prompt.format_prompt(
            text=f"{results}"
        ).to_messages()
    )

    print(answer.content)
    return answer.content


def prospecting_products(company):
    search = GoogleSerperAPIWrapper()
    results = search.results(f"{company} company products or services")

    chat = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=config['openai_api-key'])

    template = (
        "You are a helpful assistant that takes in a company knowledgeGraph and converts it to a list of products and "
        "overview of the company products or services from the view of a sales development rep who is trying to "
        "secure a meeting with someone at the company. KnowledgeGraph: {text}."
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    # get a chat completion from the formatted messages
    answer = chat(
        chat_prompt.format_prompt(
            text=f"{results}"
        ).to_messages()
    )

    print(answer.content)
    return answer.content


def prospecting_market(company):
    search = GoogleSerperAPIWrapper()
    results = search.results(f"{company} company target market")

    chat = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=config['openai_api-key'])

    template = (
        "You are a helpful assistant that takes in a company knowledgeGraph and converts it to an overview of the "
        "company's target market and audience. KnowledgeGraph: {text}."
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    # get a chat completion from the formatted messages
    answer = chat(
        chat_prompt.format_prompt(
            text=f"{results}"
        ).to_messages()
    )

    print(answer.content)
    return answer.content


def prospecting_achievements(company):
    search = GoogleSerperAPIWrapper()
    results = search.results(f"{company} company target market")

    chat = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=config['openai_api-key'])

    template = (
        "You are a helpful assistant that takes in a company knowledgeGraph and converts it to an overview of the "
        "company's recent achievements or accomplishments. KnowledgeGraph: {text}."
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    # get a chat completion from the formatted messages
    answer = chat(
        chat_prompt.format_prompt(
            text=f"{results}"
        ).to_messages()
    )

    print(answer.content)
    return answer.content


def prospecting_industry_pain(company):
    search = GoogleSerperAPIWrapper()
    results = search.results(f"Major challenges faced by companies in {company} industry in 2023")

    chat = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=config['openai_api-key'])

    template = (
        "You are a helpful assistant that takes in a company knowledgeGraph and converts it to an overview of the "
        "pain points or challenges faced by companies in the same industry."
        "KnowledgeGraph: {text}."
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    # get a chat completion from the formatted messages
    answer = chat(
        chat_prompt.format_prompt(
            text=f"{results}"
        ).to_messages()
    )

    print(answer.content)
    return answer.content


def prospecting_concerns(company):
    search = GoogleSerperAPIWrapper()
    results = search.results(f"Top concerns for CEOs in {company}'s industry")

    chat = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=config['openai_api-key'])

    template = (
        "You are a helpful assistant that takes in a company knowledgeGraph and converts it to an overview of the "
        "concerns by CEOs in the same industry."
        "KnowledgeGraph: {text}."
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    # get a chat completion from the formatted messages
    answer = chat(
        chat_prompt.format_prompt(
            text=f"{results}"
        ).to_messages()
    )

    print(answer.content)
    return answer.content


def prospecting_operational_challenges(company):
    search = GoogleSerperAPIWrapper()
    results = search.results(f"Operational challenges in {company}'s industry")

    chat = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=config['openai_api-key'])

    template = (
        "You are a helpful assistant that takes in a company knowledgeGraph and converts it to an overview of the "
        "operational challenges for all companies in the same industry."
        "KnowledgeGraph: {text}."
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    # get a chat completion from the formatted messages
    answer = chat(
        chat_prompt.format_prompt(
            text=f"{results}"
        ).to_messages()
    )

    print(answer.content)
    return answer.content


def prospecting_latest_news(company):
    search = GoogleSerperAPIWrapper()
    results = search.results(f"{company} latest news or press releases")

    chat = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=config['openai_api-key'])

    template = (
        "You are a helpful assistant that takes in a company knowledgeGraph and converts it to an overview of the "
        "latest news or press releases for the company."
        "KnowledgeGraph: {text}."
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_message_prompt = HumanMessagePromptTemplate.from_template("{text}")

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    # get a chat completion from the formatted messages
    answer = chat(
        chat_prompt.format_prompt(
            text=f"{results}"
        ).to_messages()
    )

    print(answer.content)
    return answer.content


def prospecting_recent_events(company):
    search = GoogleSerperAPIWrapper()
    results = search.results(f"{company} participation in recent industry events or webinars")

    chat = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=config['openai_api-key'])

    template = (
        "You are a helpful assistant that takes in a company knowledgeGraph and converts it to an overview of the "
        "company's participation in recent industry events or webinars."
        "KnowledgeGraph: {text}."
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_message_prompt = HumanMessagePromptTemplate.from_template("{text}")

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    # get a chat completion from the formatted messages
    answer = chat(
        chat_prompt.format_prompt(
            text=f"{results}"
        ).to_messages()
    )

    print(answer.content)
    return answer.content


def prospecting_customer_feedback(company):
    search = GoogleSerperAPIWrapper()
    results = search.results(f"Customer testimonials or feedback for {company}")

    chat = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=config['openai_api-key'])

    template = (
        "You are a helpful assistant that takes in a company knowledgeGraph and converts it to an overview of the "
        "customer testimonials or feedback for the company."
        "KnowledgeGraph: {text}."
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_message_prompt = HumanMessagePromptTemplate.from_template("{text}")

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    # get a chat completion from the formatted messages
    answer = chat(
        chat_prompt.format_prompt(
            text=f"{results}"
        ).to_messages()
    )

    print(answer.content)
    return answer.content


def prospecting_recent_partnerships(company):
    search = GoogleSerperAPIWrapper()
    results = search.results(f"Recent partnerships, mergers, or acquisitions involving {company}")

    chat = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=config['openai_api-key'])

    template = (
        "You are a helpful assistant that takes in a company knowledgeGraph and converts it to an overview of the "
        "recent partnerships, mergers, or acquisitions involving the company."
        "KnowledgeGraph: {text}."
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_message_prompt = HumanMessagePromptTemplate.from_template("{text}")

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    # get a chat completion from the formatted messages
    answer = chat(
        chat_prompt.format_prompt(
            text=f"{results}"
        ).to_messages()
    )

    print(answer.content)
    return answer.content

@bio_blueprint.route('/prospecting_bio', methods=['POST'])
def prospecting_bio():
    data = request.json
    company = data['company']
    overview = prospecting_overview(company)
    products = prospecting_products(company)
    market = prospecting_market(company)
    achievements = prospecting_achievements(company)

    bio = f"{overview}\n\n {products}\n\n {market}\n\n {achievements}"

    return jsonify({"bio": bio})

@bio_blueprint.route('/prospecting_challenges', methods=['POST'])
def prospecting_challenges():
    data = request.json
    company = data['company']
    industry_pain = prospecting_industry_pain(company)
    concerns = prospecting_concerns(company)
    operational_challenges = prospecting_operational_challenges(company)

    challenges = f"{industry_pain}\n\n {concerns}\n\n {operational_challenges}"

    return jsonify({"challenges": challenges})

@bio_blueprint.route('/prospecting_news', methods=['POST'])
def prospecting_news():
    data = request.json
    company = data['company']
    latest_news = prospecting_latest_news(company)
    recent_events = prospecting_recent_events(company)
    customer_feedback = prospecting_customer_feedback(company)
    recent_partnerships = prospecting_recent_partnerships(company)

    news = f"{latest_news}\n\n {recent_events}\n\n {customer_feedback}\n\n {recent_partnerships}"

    return jsonify({"news": news})






# prospecting_operational_challenges("Microsoft")

import json
import os

from flask import jsonify, request, Blueprint
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from dotenv import load_dotenv

# import main
#
# with open('config.json') as f:
#     config = json.load(f)
load_dotenv()
serper_api_key = os.getenv("SERPER_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

from langchain.utilities import GoogleSerperAPIWrapper

bio_blueprint = Blueprint('bio_blueprint', __name__)
search = GoogleSerperAPIWrapper()
chat = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k", openai_api_key=openai_api_key)


@bio_blueprint.route('/generate_bio', methods=['POST'])
def generate_bio():
    data = request.json  # Expecting the search results in JSON format in the POST request
    bio = prospecting_overview(data)

    return jsonify({"bio": bio})


def prospecting_overview(company):
    results = search.results(f"{company} company")
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
    results = search.results(f"{company} company products or services")

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
    results = search.results(f"{company} company target market")

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
    results = search.results(f"{company} company target market")

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
    results = search.results(f"Major challenges faced by companies in {company} industry in 2023")

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
    results = search.results(f"Top concerns for CEOs in {company}'s industry")

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
    results = search.results(f"Operational challenges in {company}'s industry")

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


@bio_blueprint.route('/prospecting_with_contacts', methods=['POST', 'GET'])
def prospecting_with_contact(contact_id):
    from main import Account, app, Contact, db

    with app.app_context():
        contact = Contact.query.get(contact_id)
        company = contact.AccountId
        company = Account.query.get(company).Name

        contact_dict = {}
        contact_dict[contact.Name] = contact.Title


        template = (
            "You are a helpful assistant that takes in about one employee, a company "
            "overview, company challenges, company recent news. This information is the result of a sales representative "
            "researching a potential customer. Your goal is to help provide a way in to the company by providing the "
            "sales rep a job pain point for each contact while keeping in mind the job title, company, company industry, "
            "company size.   It will be returned in the following format: 'employee_name': 'recommendation'.  Take your "
            "time and read through all the company information and think it through step by step. \n\n"
    
            "Info: {text}."
        )
        account = Account.query.get(contact.AccountId)
        if account.Overview is None:
            overview = prospecting_overview(company)
            products = prospecting_products(company)
            market = prospecting_market(company)
            achievements = prospecting_achievements(company)
        else:
            overview = Account.Overview
            products = Account.Products
            market = Account.Market
            achievements = Account.Achievements


        bio = f"{overview}\n\n {products}\n\n {market}\n\n {contact_dict}\n\n {achievements}"

        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        human_message_prompt = HumanMessagePromptTemplate.from_template("{text}")

        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )

            # get a chat completion from the formatted messages
        answer = chat(
            chat_prompt.format_prompt(
                text=f"{bio}"
            ).to_messages()
        )
        contact.Recommendation = answer.content
        db.session.add(contact)
        db.session.add(account)
        db.session.commit()

    return answer.content


@bio_blueprint.route('/prospecting_with_contacts', methods=['POST', 'GET'])
def prospecting_with_contacts(company_id='001Dp00000KBTVRIA5'):
    from main import Account, app, Contact

    with app.app_context():
        company = Account.query.get(company_id).Name
        # contacts = Account.query.get(company_id).Contacts

        # do a sql query to get the contacts for the company called test
        contacts = Contact.query.filter_by(AccountId=company_id).all()

    company_dict = {}
    for employee in contacts:
        company_dict[employee.Name] = employee.Title

    template = (
        "You are a helpful assistant that takes in a company directory of its employees and their roles, a company "
        "overview, company challenges, company recent news. This information is the result of a sales representative "
        "researching a potential customer. Your goal is to help provide a way in to the company by providing the "
        "sales rep a job pain point for each contact while keeping in mind the job title, company, company industry, "
        "company size.   It will be returned in the following format: 'company_name': 'recommendation'.  Take your "
        "time and read through all the company information and think it through step by step. \n\n"

        "Dictionary: {text}."
    )

    overview = prospecting_overview(company)
    products = prospecting_products(company)
    market = prospecting_market(company)
    achievements = prospecting_achievements(company)

    bio = f"{overview}\n\n {products}\n\n {market}\n\n {company_dict}\n\n {achievements}"

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_message_prompt = HumanMessagePromptTemplate.from_template("{text}")

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    # get a chat completion from the formatted messages
    answer = chat(
        chat_prompt.format_prompt(
            text=f"{bio}"
        ).to_messages()
    )

    return answer.content


def prospecting_latest_news(company):
    results = search.results(f"{company} latest news or press releases")

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
    results = search.results(f"{company} participation in recent industry events or webinars")

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
    results = search.results(f"Customer testimonials or feedback for {company}")

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
    results = search.results(f"Recent partnerships, mergers, or acquisitions involving {company}")

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

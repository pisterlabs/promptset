from bs4 import BeautifulSoup
import requests
import json
from langchain.schema import SystemMessage
import streamlit as st
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import os
import pandas as pd
from langchain.schema import SystemMessage

load_dotenv()

browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
proxycurl_api_key = os.getenv("PROXYCURL_API_KEY")

# Scraping function for Website URL
def scrape_website_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # will raise an HTTPError if the HTTP request returned an unsuccessful status code
        soup = BeautifulSoup(response.content, 'html.parser')
        # Assuming the main content is within the 'main' tag, adjust as needed for the website structure
        main_content = soup.find('body')
        return main_content.text
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None
    
# 3. Tool fürs Zusammenfassen
def summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output

# Combine summaries into one text
def combine_summaries(website_summary, company_summary):
    combined_summary = f"{website_summary}\n\n{company_summary}"
    return combined_summary

# 5. E-Mail generierung
def generate_linkedIn_email(prompt):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    # Encapsulate the prompt in a SystemMessage object, and put it in a list of lists as required
    message_objects = [[SystemMessage(content=prompt)]]

    # Log the prompt
    print("Generating email with prompt:", prompt)

    # Encapsulate the prompt in a SystemMessage object, and put it in a list of lists as required
    message_objects = [[SystemMessage(content=prompt)]]

    try:
        # The generate method now receives the correct format of messages
        result = llm.generate(message_objects, max_tokens=600)
        # Extract the content from the resulting ChatGeneration objects
        return [generation.message.content for generation in result.generations[0]]
    except AttributeError as e:
        print(f"An error occurred with message: {e}")
        # Potentially log the full traceback here
        raise

# Function to get the linkedin_profile
def get_profile(profile_id):
    api_endpoint = 'https://nubela.co/proxycurl/api/v2/linkedin'
    header_dic = {'Authorization': 'Bearer ' + proxycurl_api_key}
    params = {
        'url': f'{profile_id}',
        'extra': 'include',
        'inferred_salary': 'include',
        'skills': 'include',
        'use_cache': 'if-present',
        'fallback_to_cache': 'on-error',
        }
    response = requests.get(api_endpoint,
                            params=params,
                            headers=header_dic)
    return response.json()

def get_company(company_id):
    api_endpoint = 'https://nubela.co/proxycurl/api/linkedin/company'
    headers = {'Authorization': 'Bearer ' + proxycurl_api_key}
    params = {
        'url': f'{company_id}',
        'resolve_numeric_id': 'true',
        'categories': 'include',
        'funding_data': 'include',
        'extra': 'include',
        'exit_data': 'include',
        'acquisitions': 'include',
        'use_cache': 'if-present',
    }
    response = requests.get(api_endpoint, params=params, headers=headers)
    
    # Check if the response status code is 200 (OK)
    if response.status_code == 200:
        try:
            return response.json()
        except json.decoder.JSONDecodeError:
            print("Response is not in JSON format. Response text is:", response.text)
            return None
    else:
        print(f"Failed to fetch data: {response.status_code}")
        return None
    

# Hilfsfunktionen zur Extraktion von Profilinformationen
def extract_experience(experiences):
    experience_summary = ""
    for exp in experiences:
        exp_description = exp.get('description', 'Keine Beschreibung verfügbar.')
        exp_summary = f"{exp['title']} bei {exp['company']} ({exp['location']}): {exp_description}. "
        experience_summary += exp_summary
    return experience_summary

def extract_education(education):
    education_summary = ""
    for edu in education:
        edu_description = edu.get('description', 'Keine Beschreibung verfügbar.')
        edu_summary = f"{edu['degree_name']} in {edu['field_of_study']} an der {edu['school']}: {edu_description}. "
        education_summary += edu_summary
    return education_summary

def extract_certifications(certifications):
    cert_summary = ""
    for cert in certifications:
        cert_summary += f"{cert['name']} von {cert.get('authority', 'Unbekannte Autorität')}. "
    return cert_summary


# Streamlit Interface
st.title("LinkedIn Scraper & LLM Text Generation")

company_services = st.text_input("Unsere Services")

uploaded_file = st.file_uploader("CSV-Datei hochladen", type=["csv"])
if uploaded_file is not None:
    # CSV-Datei einlesen
    data = pd.read_csv(uploaded_file)

    if st.button("Start"):
        generated_emails_list = []

        for index, row in data.iterrows():
            # Extrahieren der URLs aus der aktuellen Zeile
            company_url = row['Unternehmenswebseite']
            company_linkedin_url = row['LinkedIn-UnternehmensURL']
            linkedin_url = row['LinkedIn-PersonenURL']


            # Initialize summaries with default values
            website_summary = "No website summary available."
            person_summary = "No person summary available."
            company_linkedin_summary = "No company LinkedIn summary available."

            # Scraping website content
            website_content = scrape_website_content(company_url)
            if website_content:
                website_summary = summary("summarize for business insights", website_content)
            else:
                st.error("Failed to scrape the website content.")
    
            # Get the company and person content    
            print("Starting to scrape the company linkedin url")
            company_linkedin_content = get_company(company_linkedin_url)
            print("Starting to scrape the person linkedin url")
            person_content = get_profile(linkedin_url)

            profile_summary = ""
            if 'headline' in person_content:
                profile_summary += f"Aktuelle Position: {person_content['headline']}. "
            if 'experiences' in person_content:
                profile_summary += extract_experience(person_content['experiences'])
            if 'education' in person_content:
                profile_summary += extract_education(person_content['education'])
            if 'certifications' in person_content:
                profile_summary += extract_certifications(person_content['certifications'])
            if 'full_name' in person_content:
                profile_summary += f"Name: {person_content['full_name']}. "

            person_summary = summary("summarize for business insights", profile_summary)
        
            # Get the company summary
            if 'description' in company_linkedin_content:
                company_linkedin_summary = summary("summarize for business insights", company_linkedin_content['description'])
            else:
            # Handle the absence of 'description', perhaps by logging an error or using a placeholder
                company_linkedin_summary = "Description not available."

    
        # Combine the summaries
            print("Combining the summaries")
            combined_summary = combine_summaries(website_summary, company_linkedin_summary)

        # Prepare the email content with actual values instead of placeholders
            email_prompt_template = """
E-Mail-Erstellung für Social Selling auf LinkedIn

Kontext: Sie erstellen eine personalisierte E-Mail, um {company_services} an einen potenziellen Kunden vorzustellen. Diese E-Mail sollte speziell darauf ausgerichtet sein, {person_summary} anzusprechen, die {combined_summary} repräsentiert.

Anweisungen:

Unternehmensdienstleistungen: Beschreiben Sie die Schlüsselangebote und einzigartigen Verkaufsargumente von {company_services}. Dies bildet die Grundlage des Verkaufsgesprächs.
Empfängerprofil (Personenzusammenfassung): Verwenden Sie {person_summary}, um die E-Mail zu personalisieren. Erwähnen Sie relevante Erfahrungen, Rollen oder Interessen der Person, um die Verbindung bedeutungsvoller zu gestalten.
Zielunternehmensprofil: Detaillieren Sie {combined_summary}, um die angebotenen Dienstleistungen mit den Bedürfnissen und Herausforderungen des Zielunternehmens in Einklang zu bringen.
E-Mail-Struktur:

Einleitung: Stellen Sie sich und Ihr Unternehmen kurz vor.
Personalisierung: Verbinden Sie den Hintergrund des Empfängers oder die Bedürfnisse des Unternehmens mit Ihren Dienstleistungen.
Wertangebot: Stellen Sie klar dar, wie Ihre Dienstleistungen ihrem Geschäft zugutekommen können.
Potenzielle Zusammenarbeit: Kreire drei Möglichkeiten, wie "{company_services}" dem Unternehmen helfen kann.
Handlungsaufforderung: Fügen Sie einen klaren nächsten Schritt hinzu, wie das Vereinbaren eines Treffens oder eines Anrufs zur weiteren Diskussion. Fügen sie dazu einen calendly link ein.
Schluss: Beenden Sie mit einem professionellen und freundlichen Abschluss.
Ton und Stil: Die E-Mail sollte professionell und dennoch zugänglich, überzeugend, aber nicht aggressiv, sowie kurz, aber informativ sein.

Compliance-Erinnerung: Stellen Sie sicher, dass die Nachricht den Richtlinien von LinkedIn und allen relevanten Gesetzen zur Marketingkommunikation entspricht.

Vermeiden sie die Formulierung "ich hoffe, diese E-Mail findet Sie gut.", weil die Formulierung nur im Englischen funktioniert.
Beschränke die Mail auf Maximal 600 Tokens.

Generieren Sie die E-Mail:
"""

# Replace placeholders with actual content
            email_prompt = email_prompt_template.format(
                company_services=company_services,
                combined_summary=combined_summary,
                person_summary=person_summary
        )

# Generate the email with the combined summary
            print(email_prompt)
            generated_email = generate_linkedIn_email(email_prompt)
            generated_emails_list.append(generated_email)

        # Display the generated emails
        for email in generated_emails_list:
            st.subheader("Generierte E-Mail:")
            st.write(email)


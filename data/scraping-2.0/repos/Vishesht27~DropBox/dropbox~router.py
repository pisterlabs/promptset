import os
import base64
import requests
from datetime import date

from flask import render_template, request, redirect, url_for, flash
from dropbox import app

from dotenv import load_dotenv
load_dotenv()

import openai
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

api_key = os.environ.get("DROPBOX_API_KEY")
AZURE_TEXT_ANALYTICS_ENDPOINT = os.environ.get("AZURE_TEXT_ANALYTICS_ENDPOINT")
AZURE_TEXT_ANALYTICS_KEY = os.environ.get("AZURE_TEXT_ANALYTICS_KEY")
openai.api_key = os.environ.get("OPENAI_API_KEY")


def extract_key_phrases(text):
    endpoint = (
        f"{AZURE_TEXT_ANALYTICS_ENDPOINT}text/analytics/v3.1-preview.4/keyPhrases"
    )
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_TEXT_ANALYTICS_KEY,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    data = {"documents": [{"id": "1", "text": text}]}

    response = requests.post(endpoint, headers=headers, json=data)
    if response.status_code == 200:
        phrases = response.json()["documents"][0]["keyPhrases"]
        return phrases
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return []


STANDARD_CLAUSES = {
    "employment": ["non-compete", "confidentiality", "benefits", "termination"],
    "rental": ["maintenance", "deposit", "termination", "renewal"],
    "nda": ["disclosure", "duration", "penalties"],
}


def generate_recommendations(contract_type, key_phrases):
    missing_clauses = [
        clause
        for clause in STANDARD_CLAUSES[contract_type]
        if clause not in key_phrases
    ]
    recommendations = []
    for clause in missing_clauses:
        recommendations.append(f"Consider adding a {clause} clause.")
    return recommendations


def enhance_sentence_with_gpt(sentence):
    response = openai.Completion.create(
        engine="davinci",
        prompt=f"Given the provided employment conditions, rewrite them in a clear, concise, and legally appropriate manner: {sentence}",
        max_tokens=150,
    )
    return response.choices[0].text.strip().replace("Rephrased:", "").strip()


def authenticate_client():
    ta_credential = AzureKeyCredential(AZURE_TEXT_ANALYTICS_KEY)
    text_analytics_client = TextAnalyticsClient(
        endpoint=AZURE_TEXT_ANALYTICS_ENDPOINT, credential=ta_credential
    )
    return text_analytics_client


client = authenticate_client()


def send_to_dropbox_sign(contract_content, api_key, email_address):
    url = "https://api.hellosign.com/v3/signature_request/send"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{api_key}:'.encode()).decode()}"
    }

    # Prepare data for multi-part form upload
    data = {
        "test_mode": "1",
        "title": "Test Contract",
        "subject": "Please sign this contract",
        "message": "This is a test contract. Please sign it.",
        "signers[0][name]": "Vishesh Tripathi",
        "signers[0][email_address]": email_address,
    }

    # Attach the file
    files = {
        "file[0]": (
            "contract.txt",
            contract_content,
        )  # Assuming contract_content is plain text. Adjust as needed.
    }

    response = requests.post(url, headers=headers, data=data, files=files)

    if response.status_code != 200:
        raise Exception(
            f"Failed to send contract to Dropbox Sign. Error: {response.json()}"
        )

    return "Contract sent successfully to Dropbox Sign!"


@app.route("/")
def index():
    return render_template("select_contract.html")


@app.route("/select_contract", methods=["GET"])
def select_contract():
    return render_template("select_contract.html")


@app.route("/generate_contract", methods=["POST"])
def generate_contract():
    contract_type = request.form["contract_type"]
    if contract_type == "rental":
        return render_template("rental_form.html")
    elif contract_type == "nda":
        return render_template("nda_form.html")
    elif contract_type == "employment":
        return render_template("employment_form.html")


def analyze_sentiment_with_azure(clause):
    response = client.analyze_sentiment(documents=[clause])[0]
    sentiment = response.sentiment
    if sentiment == "positive":
        return "Positive tone detected in the custom clause."
    elif sentiment == "neutral":
        return "Neutral tone detected in the custom clause."
    else:
        return "Negative tone detected in the custom clause."


@app.route("/finalize_contract", methods=["POST"])
def finalize_contract():
    contract_type = request.form.get("contract_type", "")
    custom_clause = request.form.get("custom_clause", "")
    sentiment_feedback = analyze_sentiment_with_azure(custom_clause)

    if contract_type == "rental":
        landlord_name = request.form.get("landlord_name", "")
        tenant_name = request.form.get("tenant_name", "")
        property_address = request.form.get("property_address", "")
        rent_amount = request.form.get("rent_amount", "")
        duration = request.form.get("duration", "")

        with open("dropbox/contract_templates/rental.txt", "r") as file:
            contract = file.read()
            contract = contract.replace("[LANDLORD_NAME]", landlord_name)
            contract = contract.replace("[TENANT_NAME]", tenant_name)
            contract = contract.replace("[PROPERTY_ADDRESS]", property_address)
            contract = contract.replace("[RENT_AMOUNT]", rent_amount)
            contract = contract.replace("[DURATION]", duration)

    elif contract_type == "nda":
        party_one = request.form.get("disclosing_party", "")
        party_two = request.form.get("receiving_party", "")
        duration = request.form.get("duration", "")

        with open("dropbox/contract_templates/nda.txt", "r") as file:
            contract = file.read()
            contract = contract.replace("[PARTY_ONE_NAME]", party_one)
            contract = contract.replace("[PARTY_TWO_NAME]", party_two)
            contract = contract.replace("[DURATION]", duration)

    elif contract_type == "employment":
        employer_name = request.form.get("employer_name", "")
        employee_name = request.form.get("employee_name", "")
        position = request.form.get("Position", "")
        salary = request.form.get("salary", "")
        duration = request.form.get("duration", "")

        with open("dropbox/contract_templates/employment.txt", "r") as file:
            contract = file.read()
            contract = contract.replace("[EMPLOYER_NAME]", employer_name)
            contract = contract.replace("[EMPLOYEE_NAME]", employee_name)
            contract = contract.replace("[POSITION]", position)
            contract = contract.replace("[SALARY]", salary)
            contract = contract.replace("[DURATION]", duration)
            # contract = contract.replace("[DATE]", str(date.today()))

    ai_enhance = request.form.get("ai_enhance")
    if ai_enhance == "yes":
        custom_clause = enhance_sentence_with_gpt(custom_clause)
        sentiment_feedback = analyze_sentiment_with_azure(custom_clause)

    # Split the custom clause into individual terms
    custom_clause_list = [
        clause.strip() for clause in custom_clause.split(".") if clause
    ]

    # Format the terms in a point-wise manner
    custom_clause_formatted = "\n".join("- " + clause for clause in custom_clause_list)

    # Construct the full contract
    full_contract = f"{contract}\n\nTerms and Conditions:\n{custom_clause_formatted}"

    # Extract key phrases
    key_phrases = extract_key_phrases(contract)

    # Generate recommendations
    recommendations = generate_recommendations(contract_type, key_phrases)

    return render_template(
        "output_template.html",
        contract=full_contract,  # Use the full contract here
        recommendations=recommendations,
        custom_clause=custom_clause,
        sentiment_feedback=sentiment_feedback,
    )


@app.route("/send_to_dropbox", methods=["POST"])
def send_to_dropbox():
    full_contract = request.form["full_contract"]
    message = send_to_dropbox_sign(full_contract, api_key, request.form['signing_email'])
    return render_template("success_notification.html", message=message)

if __name__ == "__main__":
    app.run(debug=True)
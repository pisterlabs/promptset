#Nolan Blevins

import requests
from bs4 import BeautifulSoup
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime
import PyPDF2
import openai
import configparser
import os
from fpdf import FPDF

# Load configuration file
config = configparser.ConfigParser()
config.read("config.ini")

# Set up your GPT-3 API key
openai.api_key = config.get("openai", "api_key")


def parse_xml_to_dataframe(xml_file):
    # Load XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Define the namespace
    namespace = {"ns": "http://apply.grants.gov/system/OpportunityDetail-V1.0"}

    # Define a list to store the data dictionaries
    data_list = []

    # Iterate over each OpportunitySynopsisDetail_1_0 element
    for opportunity in root.findall("ns:OpportunitySynopsisDetail_1_0", namespace):
        # Define a dictionary to store the data for each entry
        data = {}

        # Iterate over each child element of the opportunity
        for subchild in opportunity:
            tag = subchild.tag.replace(
                "{http://apply.grants.gov/system/OpportunityDetail-V1.0}", ""
            )
            data[tag] = subchild.text

        # Append the data dictionary to the list
        data_list.append(data)

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data_list)
    return df


def remove_entries_with_past_close_date(df):
    # Get the current date
    today = datetime.now().date()

    # Convert CloseDate column to datetime with the specified format
    df["CloseDate"] = pd.to_datetime(df["CloseDate"], format="%m%d%Y")

    # Filter the DataFrame to remove entries with CloseDate before today
    filtered_df = df[df["CloseDate"].dt.date >= today]
    return filtered_df


def remove_entries_with_invalid_url(df):
    # Filter the DataFrame to remove entries where the AdditionalInformationURL does not end with ".html" or ".pdf"
    filtered_df = df[
        df["AdditionalInformationURL"].str.endswith((".html", ".pdf"), na=False)
    ]
    return filtered_df


def extract_text_from_html(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        return text
    except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
        print(f"Error retrieving HTML content from {url}: {str(e)}")
        return ""


def extract_text_from_pdf(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open("temp.pdf", "wb") as f:
            f.write(response.content)

        with open("temp.pdf", "rb") as f:
            reader = PyPDF2.PdfFileReader(f)
            text = ""
            for page in range(reader.numPages):
                text += reader.getPage(page).extractText()

        return text
    except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
        print(f"Error retrieving PDF content from {url}: {str(e)}")
        return ""
    except PyPDF2.PdfReadError as e:
        print(f"Error parsing PDF file from {url}: {str(e)}")
        return ""


# GPT-3.5 Turbo API call function to generate dialogue
def generate_grant_proposal():
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a Grant proposal writer for a non-profit organization."},
            {
                "role": "user",
                "content": "Can you write a grant proposal using the information from the provided files?",
            },
            {"role": "system", "content": text[:2000]},
        ],
    )
    return completion.choices[0].message.content.strip()


# Parse the XML file into a DataFrame
df = parse_xml_to_dataframe("Grants_may18_xml.xml")

# Remove entries with CloseDate before today
filtered_df = remove_entries_with_past_close_date(df)

# Remove entries with invalid AdditionalInformationURL
filtered_df = remove_entries_with_invalid_url(filtered_df)

# Export the filtered DataFrame to a CSV file
filtered_df.to_csv("filtered_data.csv", index=False)


# Select the row at the 3rd index from the filtered DataFrame
row = filtered_df.iloc[40]

# Extract the URL from the selected row
url = row["AdditionalInformationURL"]

# Check if the URL ends with ".html" or ".pdf" and extract the text accordingly
if url.endswith(".html"):
    text = extract_text_from_html(url)
elif url.endswith(".pdf"):
    text = extract_text_from_pdf(url)
else:
    print(f"Invalid URL format: {url}")
    text = ""

# Save the extracted text to a file
with open("extracted_text.txt", "w") as file:
    file.write(text)


def save_grant_proposal_to_pdf(grant_proposal):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Split grant proposal into lines
    lines = grant_proposal.split("\n")

    # Output each line as a separate cell
    for line in lines:
        pdf.multi_cell(0, 10, line)

    pdf.output("grant_proposal.pdf")


# Generate the grant proposal using the extracted text
grant_proposal = generate_grant_proposal()

# Save the grant proposal to a PDF file
save_grant_proposal_to_pdf(grant_proposal)

# Print the generated grant proposal
print(grant_proposal)


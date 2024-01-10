import argparse
import pandas as pd
from openai import OpenAI
import time
from get_endpoint_html import setup_driver, get_page_text
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import streamlit as st
import ssl


class ComplimentGenerator:
    def __init__(self, csv_file, api_key):
        self.csv_file = csv_file

        self.driver = setup_driver()
        self.client = OpenAI(api_key=api_key)
        self.website_column = "site"
        self.modified_prompt = """Using the source text from the business's website: '''{page_text}''', craft a unique and personalized compliment that highlights a specific and distinctive aspect or feature evident in the text. This compliment should reflect an intimate understanding of what sets the business apart, focusing on elements such as a particular product, service, or an aspect of their customer experience, as gleaned from the page text. It should be concise , and serve as an effective segue into the offer of a complimentary website audit. The compliment should feel genuine and be clearly tailored to the unique characteristics of the business, demonstrating that it's coming from someone who has taken the time to understand and appreciate what makes the business special. The goal is to create a compliment that resonates with the specific details and qualities presented in the '''{page_text}'''. focus solely on the compliment itself. IMPORTANT NOTE: the unique compliment should be 2 lines long and the response to this prompt should only include the unique compliment. The tone of the compliment should be the same as proffessional cold emailer Ryan Deiss and should be in the second person so that it is addressing the business directly. Make sure the unique compliment is only 2 lines long, it has to be brief and an easy read! DO NOT OFFER THEM AN AUDIT, ONLY RESPOND WITH THE UNIQUE COMPLIMENT AND NOTHING ELSE, I can't stress how much you should only respond with the compliment and nothing else in the response to this prompt and make the compliment as unique as it can be towards the company but also short."""

    def fetch_page_text(self, url):
        return get_page_text(self.driver, url)

    def generate_compliment(self, page_text):
        messages = self.modified_prompt.format(page_text=page_text)

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": messages},
            ],
        )

        print(response.choices[0].message.content)
        return response.choices[0].message.content

    def process_websites(self):
        spreadsheet = self.csv_file

        website_links = list(spreadsheet[self.website_column])

        # Initialize a list to store compliments
        compliments = []

        # Loop over each website link and generate a compliment
        for website_link in website_links:
            try:
                page_text = self.fetch_page_text(website_link)
            except:
                page_text = None
            if page_text:
                compliment = self.generate_compliment(page_text)
                compliments.append(compliment)
            else:
                compliments.append(False)

        # Add the compliments as a new column to the dataframe
        spreadsheet["compliment"] = compliments
        return spreadsheet

    def send_results_email(self, dataframe, recipient_email):
        # Convert DataFrame to CSV
        filename = "compliments.csv"
        dataframe.to_csv(filename, index=False)

        # Set up the email server
        context = ssl.create_default_context()

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            sender_email = st.secrets["gmail_email"]
            sender_password = st.secrets["gmail_password"]
            # Login to the email server
            server.login(sender_email, sender_password)

            # Create the email
            msg = MIMEMultipart()
            msg["From"] = sender_email
            msg["To"] = recipient_email
            msg["Subject"] = "Here's your results for compliment generation"

            # Email body
            body = "Please find attached the compliments generated for your websites."
            msg.attach(MIMEText(body, "plain"))

            # Attach the CSV file
            attachment = open(filename, "rb")
            part = MIMEBase("application", "octet-stream")
            part.set_payload((attachment).read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition", "attachment; filename= %s" % filename
            )
            msg.attach(part)

            # Send the email
            server.send_message(msg)

        print(f"Email sent successfully to {recipient_email}")

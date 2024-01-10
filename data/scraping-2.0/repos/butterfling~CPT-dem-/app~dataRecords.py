import openai
from authenticateDatabase import connect_to_sql_database
import os

# Database connection and cursor creation
connect = connect_to_sql_database("https://cptdatabasecredentials.vault.azure.net/","dbadmin")
cursor = connect.cursor()

# Constants
table_name = "consentGiven"
email_field_name = "[Email]"  
ssn_field_name = "[Social-Security-Number]"
financial_information_field_name = "[Financial-Information]"

# Functions for database queries
def count_total_records(cursor, table_name):
    query = f"SELECT COUNT(*) FROM {table_name}"
    cursor.execute(query)
    result = cursor.fetchone()
    return result[0] if result else 0

def count_non_encrypted_emails(cursor, table_name):
    query = f"SELECT COUNT(*) FROM {table_name} WHERE {email_field_name} NOT LIKE 'b''gAAA%'"
    cursor.execute(query)
    result = cursor.fetchone()
    return result[0] if result else 0

def count_non_encrypted_social_security_numbers(cursor, table_name):
    query = f"SELECT COUNT(*) FROM {table_name} WHERE {ssn_field_name} NOT LIKE 'b''gAAA%'"
    cursor.execute(query)
    result = cursor.fetchone()
    return result[0] if result else 0

def count_non_encrypted_financial_information(cursor, table_name):
    query = f"SELECT COUNT(*) FROM {table_name} WHERE {financial_information_field_name} NOT LIKE 'b''gAAA%'"
    cursor.execute(query)
    result = cursor.fetchone()
    return result[0] if result else 0

# Gathering data
total_records = count_total_records(cursor, table_name)
encrypted_emails = count_non_encrypted_emails(cursor, table_name)
encrypted_social_security_numbers = count_non_encrypted_social_security_numbers(cursor, table_name)
financial_information = count_non_encrypted_financial_information(cursor, table_name)

data = f"total_records : {total_records}, Number of encrypted_emails in the database: {encrypted_emails}, encrypted_social_security_numbers: {encrypted_social_security_numbers} , count_non_encrypted_financial_information: {financial_information}"

# OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to get summary using OpenAI
def summarize_with_openai(data):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=(
            "You are a humorous data protection officer. Analyze the following data, which is a collection of customer records where consent has not been given according to GDPR. Provide a humorous summary, suggest potential changes for compliance, and advise on precautions to take if these records are stored in plain text.Make sure to include the numbers in your response. Here's the data:\n\n" + data
        ),
        temperature=0.7,
        max_tokens=150
    )
    return response.choices[0].text.strip()


def result():
    return summarize_with_openai(data)


# Writing summary to a file
with open('summary.txt', 'w') as file:
    file.write(result())

print("Summary written to summary.txt")

import pandas as pd
import openai
from dotenv import load_dotenv
import os
import sys

if len(sys.argv) < 2:
    print("Error: No file path provided")
    sys.exit(1)

file_path = sys.argv[1]

# Load the environment variables
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

def categorize_business_with_gpt(business_name, transaction_type, payer, categories):
    # Updated prompt to include transaction type
    prompt = (
    f"'{payer}' is the payer, '{transaction_type}' is the transaction type, and '{business_name}' is the receiver. "
    f"This is a bank statement from a user where you can see their purchases, transfers, etc. "
    f"To categorize a transaction, consider transactions with both a name and surname in both the receiver and payer fields. "
    f"Determine the most appropriate category for this transaction from the following list: {', '.join(categories)}. Answer in just one of the categories from the list."
)


    try:
        response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",  # Use the latest available model
            prompt=prompt,
            max_tokens=60
        )
        return response.choices[0].text.strip()
    except openai.error.OpenAIError as e:
        print(f"An error occurred: {e}")
        return "Error: Unable to categorize"

def categorize_transaction(row, categories):
    business_name = row['Saajan nimi']
    transaction_type = row['Tapahtumalaji']
    payer = row['Maksaja']
    return categorize_business_with_gpt(business_name, transaction_type, payer, categories)

pd.set_option('display.max_rows', 100)
# Load the CSV file
df = pd.read_csv(file_path, delimiter=';')  # Adjust delimiter if needed
transactions = df[['Summa', 'Maksupäivä', 'Maksaja', 'Saajan nimi', 'Tapahtumalaji']]  # Ensure 'Tapahtumalaji' is included

# Define your categories
categories = [
    "Groceries",
    "Utilities",
    "Entertainment",
    "Dining and Restaurants",
    "Transportation",
    "Healthcare",
    "Clothing and Apparel",
    "Technology and Electronics",
    "Subscriptions and Memberships",
    "Home and Garden",
    "Education",
    "Travel and Accommodation",
    "Gifts and Donations",
    "Financial Services",
    "Sports and Recreation",
    "Housing and Leasing",
    "Transfers",
    "Taxi",
]

# Apply categorization
transactions['Category'] = transactions.apply(lambda row: categorize_transaction(row, categories), axis=1)

# Output the categorized data
print(transactions)  # For testing, show first few rows

# Optionally, save to JSON
# transactions.to_json('/mnt/data/categorized_transactions.json')

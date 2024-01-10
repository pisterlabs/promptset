"""
Script Name: Ai summary for seller Payouts
Version: 1.0
Last Modified: 2023-06-12
Author: Ameed Jamous

Description:
This script fetches payment history from Telecomsxchange's Seller Pay History API, 
summarizes the total number of transactions and the total payout amount, 
and then uses OpenAI to generate a textual summary and identify any patterns in the data.
The script logs all its actions, both in the console and in a file named 'payhistory_summary.log'.

Credentials, API endpoints, and other configurations can be adjusted in the respective sections of the 
script.
"""

import requests
from requests.auth import HTTPDigestAuth
import openai
import json
import logging
import os
from datetime import datetime

# Set up logging
logging.basicConfig(filename='payhistory_summary.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s:%(message)s')

# You'll need to set the environment variables TCXC_USERNAME, TCXC_PASSWORD, and OPENAI_KEY 
# in your operating system or deployment environment for the script to run successfully.

# Telecomsxchange/NeuTrafix Seller Credentials
username = os.environ.get('TCXC_USERNAME')
password = os.environ.get('TCXC_PASSWORD')

# Set your OpenAI key
openai.api_key = os.environ.get('OPENAI_KEY')

# TCXC API Endpoint 
pay_history_url = 'https://apiv2.telecomsxchange.com/sellers/payhistory'

# Data for the pay history API request
pay_history_data = {
    'date_from': "2019-01-01 00:00:00",
    'date_to': "2023-06-01 00:00:00",
    'pager': "1000"
}

try:
    # Send POST request for pay history
    pay_history_response = requests.post(pay_history_url, data=pay_history_data, auth=HTTPDigestAuth(username, password))

    # Check if the pay history request was successful
    if pay_history_response.status_code == 200:
        pay_history_info = pay_history_response.json()
        if pay_history_info.get('status') == 'success':
            transactions = pay_history_info.get('transactions')
            # Create a summary of the transactions
            total_transactions = len(transactions)
            total_amount = sum([float(transaction['amount']) for transaction in transactions])
            # Prepare prompt for OpenAI
            prompt = f"We have a set of {total_transactions} financial transactions, with an accumulated payout of {total_amount} USD. The first few transactions are: {transactions[:5]}. Please generate a detailed summary of this data, emphasizing on any noticeable patterns or trends. Additionally, always provide the total sum of the payout values."
            # Generate the summary using OpenAI
            summary_completion = openai.Completion.create(model="text-davinci-003", prompt=prompt, max_tokens=200)
            # Print and log the generated summary
            summary = summary_completion.choices[0].text.strip()
            print(summary)
            logging.info(f"Generated summary: {summary}")
        else:
            error_message = f"Pay history request returned an error. Details: {pay_history_info}"
            print(error_message)
            logging.error(error_message)
    else:
        error_message = "Pay history request failed"
        print(error_message)
        logging.error(error_message)
except Exception as e:
    error_message = f"An error occurred: {str(e)}"
    print(error_message)
    logging.error(error_message)

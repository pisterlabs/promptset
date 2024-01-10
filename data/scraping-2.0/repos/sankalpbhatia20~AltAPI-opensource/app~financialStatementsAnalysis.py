import finnhub
import openai
import os
from .config import settings

# replace YOUR_API_KEY with your actual API key
finnhub_client = finnhub.Client(api_key= f'{settings.finnhub_token}')

def analyse_financial_statements(ticker, statement_to_be_analysed, frequency):
    # replace SYMBOL with the stock symbol of the company you want to get financial statements for (e.g., AAPL for Apple Inc.)
    financials = finnhub_client.financials_reported(symbol = ticker, freq = frequency)

    if statement_to_be_analysed == 'balance_sheet':
        analyzed_data = financials['data'][0]['report']['bs']
    elif statement_to_be_analysed == 'income_statement':
        analyzed_data = financials['data'][0]['report']['ic']
    elif statement_to_be_analysed == 'cashflow':
        analyzed_data = financials['data'][0]['report']['cf']
    else:
        return {'Error': 'Please enter a valid statement to be analysed'}

    openai.api_key = f'{settings.openai_token}'

    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Give me a summary of how " + ticker + " is doing financially looking at the following " + str(statement_to_be_analysed) + "data: " + str(analyzed_data)}
                ]
            )
    except Exception as e:
        return {'Error': str(e)}

    return ({f"{ticker}'s {statement_to_be_analysed} summary".upper() : completion.choices[0].message['content']})
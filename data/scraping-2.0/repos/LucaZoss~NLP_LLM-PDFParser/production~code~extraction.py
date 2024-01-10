import os
import json
import time
from openai import OpenAI
from typing import Dict
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path


env_path = Path(
    '/Users/lucazosso/Desktop/IE_Course/Hackathon/production/ATT85165.env')
load_dotenv(dotenv_path=env_path)

# Load your OpenAI API key
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

print("welcome to the matrix")

# COLUMNS for Pandas DF
COLUMNS = ['PDF ID', 'ISIN', 'Issuer', 'Currency',
           'Underlying(s)', 'Strike', 'Launch Date', 'Final Valuation Day', 'Maturity', 'Cap', 'Barrier']

text_example = '''Final Terms and Conditions (our ref. CE4247RAI) as of February 16th, 2022
15M Capped Bonus Certificate Plus Worst-of on DAX®, FTSE100
and IBEX 35® in USD Quanto
Issuer
BNP Paribas Issuance B.V. (S&P's A+)
Guarantor
BNP Paribas (S&P's A+ / Moody's Aa3 / Fitch AA-)
Issue Type
Certificate
Issue Amount
USD 1,600,000
Number of Certificates
1,600
Notional Amount per
Certificate (N)
1 Certificate = USD 1,000
Currency
USD Quanto
Issue Price per
Certificate
100.00%
Listing
None
Trade Date
February 15th, 2022
Strike Date
February 15th, 2022
Issue Date
March 01st, 2022
Redemption Valuation
May 15th, 2023
Date
Redemption Date
May 30th, 2023
Underlying Indices
i
Name of Underlying
Bloomberg
Indexi
Administrator
Register
Indexi
Initial
Code
1
DAX®
DAX
15412.71
STOXX Ltd.
Included
2
FTSE100
UKX
7608.92
FTSE
Included
International
Limited
3
IBEX 35®
IBEX
8718.00
SOCIEDAD
Included
DE BOLSAS
S.A.
-
Final Redemption
On the Redemption Date, the Issuer shall redeem each Certificate at the following Cash
Settlement Amount:
1) If WO IndexFinal is greater than or equal to 120% x WO IndexInitial:
N x 120%
2) If a Knock-out Event has not occurred and WO IndexFinal is less than 120% x WO
IndexInitial:
N x max
108.20%.
WO IndexFinal
WO IndexInitial
Equity Derivatives Solutions / Structured Products - Funds Of Funds /
1
Family Offices
romain.marsigny@bnpparibas.com

BNP PARIBAS
CORPORATE & INSTITUTIONAL BANKING
The bank for a changing world
3) If a Knock-out Event has occurred:
N x
WO IndexFinal
WO Index Initial
Where
WO Index is the Underlying Index with the worst performance from the Strike Date to the
Redemption Valuation Date, defined as:
" IndexInitial.
3
Index 'Final
WO IndexInitial is the official closing level of WO Index on the Strike Date.
WO IndexFinal is the official closing level of WO Index on the Redemption Valuation Date.
Indexi
Initial with i from 1 to 3 is the official closing level of the Indexi
on the Strike Date.
Indexi
Final with i from 1 to 3 is the official closing level of the Indexi
on the Redemption
Valuation Date.
Knock-out Level
DAX® - 10,788.8970 (70% of Index1
FTSE100 - 5,326.2440 (70% of Index2
Initial)
IBEX 35® - 6,102.60 (70% of Index3
Initial)
Initial)
Knock-out
Determination Day
The Redemption Valuation Date.
Knock-out Valuation
Time
Specific Scheduled Closing Time of each Underlying Index on the Redemption Valuation Date.
Knock-out Event
A Knock-out Event shall be deemed to occur if, at the Knock-out Valuation Time on the Knock-
out Determination Day, at least one Underlying Index closes at a level strictly less than its Knock-
out Level.
'''
# debut function


def extract_fields(pdf_texts: Dict[str, str]) -> pd.DataFrame:
    fields_df = pd.DataFrame(columns=COLUMNS)

    for pdf_id, text in pdf_texts.items():

        prompt = f'''Act as an expert in financial analysis, specializing in interpreting and extracting key data from financial term sheets.\
        Your task is to extract the following fields and their associated value(s), and return them in a proper JSON format witht the following keys:
        ISIN, Issuer, Currency, Underlying(s), Strike, Launch Date, Final Valuation Day, Maturity, Cap, Barrier
        In cases of missing informations for Cap and Barrier fields mark them as: NaN.\

        Use the following constraints delimited by triple backticks to extract the needed informations:
        ```
        - ISIN: Always 12 alphanumeric characters. If unclear, use any 12 alphanumeric characters in the document.
        - Issuer: Must be a bank name.
        - Currency: Must be a valid currency.
        - Underlying(s): Extract Bloomberg codes/tickers; multiple entries separated by commas.
        - Strike: Contains between two to six-digits and at least one decimals; find values close to 'Underlying(s)'.
        - Launch Date/Trade Date/Strike Date: In date format, excluding the issue date.Ensure to use the precise value as found in the input text.
        - Final Valuation Day/Redemption Valuation Date: In date format.
        - Maturity/Redemption date: In date format.
        - Cap: A number over 100; percentage close to an index.
        - Barrier/Bonus Barrier/Knock-In Barrier/Knock-Out Barrier: Percentage less than 100.
        ```\
        For clarity and accuracy, here is an example of the extracted fields and their associated values that you should produce from the the following {text_example},(Remember the output should be in JSON Format.):

        "ISIN": "XS2033997748",
        "Issuer": "BNP",
        "Currency": "USD",
        "Underlying(s)": ["DAX", "UKX", "IBEX"],
        "Strike": [15412.71, 7608.92, 8718.00],
        "Launch Date": "15.02.2022",
        "Final Valuation Day": "15.05.2023",
        "Maturity: "30.05.2023"
        "Cap": 120,
        "Barrier": 70
        \

        Apply the above process, using the provided definitions to extract the key information, Ensure 'Underlying(s)' and 'Strike' are close. For Barrier, specify the percentage value.\
        Text to extract is delimited by triple backtick:
        ```{text}```
        '''
        # "gpt-3.5-turbo-16k"
        try:
            print(f"prompting {pdf_id}")
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo-16k",
                messages=[{"role": "user", "content": prompt}],
            )
            # Extract the response text
            response = completion.choices[0].message.content

            response_dict = json.loads(response)
            response_dict['PDF ID'] = pdf_id
            # Convert the dictionary to a DataFrame
            response_df = pd.DataFrame([response_dict])

            # Concatenate with the existing DataFrame
            fields_df = pd.concat([fields_df, response_df], ignore_index=True)
            # fields_df = fields_df.concat(response_dict, ignore_index=True)
            time.sleep(1)  # Adjust the delay as required

        except Exception as e:
            print(f"Error processing {pdf_id}: {e}")

    return fields_df

# add loggings
# Be cautious with data types: use strings for text and dates, and use numbers (floats or integers) for numerical values.\

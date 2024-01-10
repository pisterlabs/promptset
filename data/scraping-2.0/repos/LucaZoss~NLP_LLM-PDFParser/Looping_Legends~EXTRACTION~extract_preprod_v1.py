from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv('OPEN_AI_KEY'))

print("environement is build")


# Getting text files from pdf previously created with OCR

# for pre-production purposes, upload from local file without going through OCR process
with open("/Users/lucazosso/Desktop/IE_Course/Hackathon/Looping_Legends/extracted/XS2021832634_extracted.txt", "r") as file:
    term_sheet_test = file.read()

# Fields to target


financial_product_terms = {
    "ISIN": "International Securities Identification Number",
    "Issuer": "The entity or organization that issues the financial product",
    "Ccy": "The currency in which the financial product is denominated or traded",
    "Underlying(s)": "The assets, indices, or securities on which the financial product's performance is based, for example: GLE, RNO FP,VOW3 GY, DAI GY. it is sometimes called bloomberg code/identifier",
    "Strike": "The strike price of the underlying and not the barrier. It must be a number and must not contain any letters. It is a financial number so it will have decimals. You can also calculate it by taking the knock-in barrier price divided by barrier level, and it cannot be 100% of Initial Level ",
    "Launch Date": "The date when the financial product is officially issued and made available for investment or trading",
    "Final Valuation Day": "The date on which the final valuation of the financial product is made",
    "Maturity": "The date on which the financial product is set to expire or mature",
    "Cap": "An upper limit on potential returns for the financial product",
    "Barrier": "A specific level or threshold that, if reached, can trigger certain events or determine the product's performance"
}

# There is a unique value for each Underlying so if there are multiple underlyings there will be multiple strike prices.
elements_to_extract_value_from = financial_product_terms.keys()

# Prompting


prompt = f'''
I want you to act as a data extraction specialist in the finance industry with special knowledge in financial term sheet.
Your task is to go through this text file {term_sheet_test} and extract the value of the corresponding key elements listed here {elements_to_extract_value_from}.
I have also included in this file {financial_product_terms} the meaning of each element to help you in your extraction.
Please keep in mind that if multiple underlyings are present then each underlying has its unique strick price.
As a result please provide a dictionary as an output format (key: value(s)).

'''

# Non-streaming:
print("----- standard request -----")
completion = client.chat.completions.create(
    model="gpt-3.5-turbo-16k",
    messages=[
        {
            "role": "user",
            "content": f"{prompt}",
        },
    ],
)
print(completion.choices[0].message.content)

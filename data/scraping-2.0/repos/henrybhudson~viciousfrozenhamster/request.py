#pip install openai

import os
import openai
openai.organization = "org-nsxsanhIbdwgW0Ndpb47S2kD"
openai.api_key = "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"

category_input = ""

def get_response():
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": f"Categorise the payment described {category_input} using one of the following categories. Just say the word. Bills, Charity, Food, Entertainment, Finances, General, Groceries, Holidays, Personal Care, Shopping, Bank Transfers, Transport"},
        ]
    )

    return response


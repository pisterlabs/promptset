from typing import List

import requests
from dotenv import load_dotenv
import os
import openai

load_dotenv()

API_KEY = os.getenv("APIKEY")
# openai.apikey = API_KEY

# openai.Answer.create()

answers_url = "https://api.openai.com/v1/answers"
header_auth = f"Bearer {API_KEY}"
header_contenttype = "application/json"

search_model = "ada"
completion_model = "davinci"

examples_context = """6.6.1.3	Decent Working Time

Violations of working time through the practices of state and non-state actors my limit the right of a person to 
decent working hours. Siemens may face several operational, financial, legal and reputational risks where these 
standards aren't met. Specific risks may arise from potential violations of working hours regulations 
project business on the part of suppliers, subcontractors or other businesses involved in the project.
Definition of the Issue
The right to decent working time is enshrined in Article 7 of the International Covenant on Economic, Social and 
Cultural Rights (ICESCR). Specifically, workers are entitled to the right to rest, leisure, reasonable limitation 
of working hours and adequate holiday allowances with pay. These conditions are linked to basic human 
rights including the right to an adequate standard of living (including adequate food, clothing and housing), the 
right to physical and mental health and the right to life.
Numerous ILO conventions provide for more detailed provisions on decent working hours laid down in the 
ICESCR. ILO convention 1 stipulates that generally working time should not exceed 48 hours per week, or 
eight hours a day, with at least one day off in every seven. ILO convention 30 states a working day cannot 
exceed ten hours. Furthermore, ILO convention 1 also states that where overtime work is allowed and required, 
the maximum number of additional hours must be fixed, and overtime must be compensated at a rate of 
no less than 25% above the normal rate of pay.

11.3.6	If and when Supplier fails to meet any agreed Delivery Date, except for reasons of Force Majeure, Supplier shall pay to Customer the amount of EUR 500 (five hundred Euros) for each set of Documentation delayed and 1% (one percent) of the sum of the purchase price of the Agreed Order and of the purchase prices of the Products and/or Services which cannot be used because of the delay per calendar day of delay, up to a maximum amount per event of 30% (thirty percent) of such price. The payment of any of these amounts or parts thereof shall not discharge Supplier of its obligations to supply the Products and/or Services or of any other liabilities or obligations under this Agreement and/or any Agreed Order and/or by law. Customer shall not be required to reserve its rights under this Article 11.3.7, in particular, when the Products or Services are received or accepted.
"""
examples = [
    [
        "Where is the right to decent working hours defined?",
        "In Article 7 of the International Covenant on Economic, Social and Cultural Rights (ICESCR)."
    ],
    [
        "How should overtime be compensated?",
        "At least 25% above the normal rate of pay."
    ],
    [
        "What risks does Siemens face when the decent working hour standards are not met?",
        "Siemens risks being faced with operational, financial, legal and reputational risks."
    ],
    [
        "What shall the Supplier pay in case of not meeting the agreed Delivery Date?",
        "EUR 500 (five hundred Euros) for each set of Documentation delayed and 1% (one percent) of the sum of the purchase price of the Agreed Order and of the purchase prices of the Products and/or Services which cannot be used because of the delay per calendar day of delay, up to a maximum amount per event of 30% (thirty percent) of such price."
    ],
    # [
    #     "What is Siemens’ minimum volume commitment?",
    #     "I don't know the answer to that question."
    # ]
]


def get_answer(question: str, paragraphs: List[str]):
    response = requests.post(
        answers_url,
        headers={
            "Authorization": header_auth,
            "Content-Type": header_contenttype
        },
        json={
            "documents": paragraphs,
            "question": question,
            "search_model": search_model,
            "model": completion_model,
            "examples_context": examples_context,
            "examples": examples,
            "max_tokens": 200,
            "stop": ["\n", "<|endoftext|>"],
            "temperature": 0
        }
    )
    return response.json()['answers'][0]


compl_model = "davinci"
completion_url = f"https://api.openai.com/v1/engines/{compl_model}/completions"


def get_completion(question: str):
    response = requests.post(
        completion_url,
        headers={
            "Authorization": header_auth,
            "Content-Type": header_contenttype
        },
        json={
            "prompt": """
            What is the german legal warranty period?
            """,
            "max_tokens": 50,
            "temperature": 0.1,
            "n": 1,
        }
    )
    return response.json()['choices'][0]['text']

import requests
import os
import firebase_admin
import json
import cohere
from icecream import ic
import unicodedata
from PDFReader import pdf_to_text



# Cohere initialization
co = cohere.Client(os.environ.get("COHERE_API_KEY"))

url = "https://api.cohere.ai/v1/chat"

def float_def(s: str, default: float) -> float:
    try:
        return float(s.strip("%"))
    except:
        return default

def extract_card_details(input_contract: str):
    message_1 = "Please list out the Company Name (TD, Royal Bank of Canada, etc), Card Name (CIBC Dividend Card, RBC Ion Plus Visa, etc), Card Type (card_type) [ie Visa, Mastercard] (str), Avg APR (avg_apr) (float) - use first APR seen, " \
            "Minimum Cashback (float percentage), Foreign Transaction Fee (float percentage), Sign up Offer (float percentage), Annual Fee (float), and finally Overcharge fee (float percentage)." \
            "Only describe one card list in a single level JSON format. If you don't have data just put N/A for strings and a '-1' for floats. " \
            "The output should look like EXAMPLE: { Bank Name: , Card Name: , Card Type: , Avg Apr: , Min Cashback: , Max Cashback: , Foreign Transaction Fee: , Sign up Offer: , Annual Fee: , Overcharge Fee: } in valid JSON."
    response_1 = co.chat(
        message=message_1,
        # documents has title of document and snippet of text
        documents=[
            {"title": "Card Terms and Conditions", "snippet": input_contract},
        ],
        prompt_truncation="AUTO",
    )

    # Get response from Cohere API and print the answer text

    response_data = response_1.text

    user_question = {
        "user_name": "User",
        "text": message_1,
    }


    # remove delimiters
    bot_answer_text_1 = (
        response_data
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    # convert to dict
    bot_answer_json = ic(json.loads(bot_answer_text_1))

    row_entry = {
        "name": bot_answer_json["Card Name"],
        "card_type": bot_answer_json["Card Type"],
        "avg_apr": float_def(bot_answer_json["Avg Apr"], float('nan')),
        "min_cashback": float_def(bot_answer_json["Min Cashback"], 0),
        "max_cashback": float_def(bot_answer_json["Max Cashback"], 0),
        "foreign_fee": float_def(bot_answer_json["Foreign Transaction Fee"], 0),
        "intro_offer_details": bot_answer_json["Sign up Offer"],
        "company_name": bot_answer_json["Bank Name"],
        "annual_fee": float_def(bot_answer_json["Annual Fee"], 0),
        "overcharge_fee": float_def(bot_answer_json["Overcharge Fee"], float('nan')),
        "contract": input_contract,
    }

    ic(row_entry)

    return row_entry

def validate_comment(answer: str) -> bool:
    l_ans = answer.lower()

    if "i don't know" in l_ans:
        return False
    if "sorry" in l_ans or "unfortunately" in l_ans:
        return False

    return True

def ask_more_card_details(input_contract: str, question: str, tries: int = 3):
    response = co.chat(
        message=question,
        # documents has title of document and snippet of text
        documents=[
            {"title": "Card Terms and Conditions", "snippet": input_contract},
        ],
        prompt_truncation="AUTO",
    )

    answer = response.text
    if not validate_comment and tries > 1:
        return ask_more_card_details(input_contract, question, tries-1)
    
    return answer



if __name__ == "__main__":
    result_text = pdf_to_text("../data/CIBC-Premium.pdf")

    print("Sanitized PDF:")
    print(result_text)

    extract_card_details(result_text)

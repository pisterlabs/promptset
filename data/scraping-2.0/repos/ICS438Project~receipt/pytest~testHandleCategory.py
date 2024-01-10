#pip install --upgrade jupyter_client testbook pytest

import pytest
from handleVendor import read_text_files, process_and_validate_json, generate_receipt_json, generate_embeddings, convert_to_embeddings_df, split_database, cosine_similarity_clf, process_list
import os
import json
import pandas as pd
from langchain.llms import OpenAI

def test_read_text_files():
    text_list = []
    # Given
    folder_path = './test file'
    with open('./test file/receipt1.txt', 'r') as file:
        expected_file = file.read()

    # When
    actual_file_names = read_text_files(folder_path)

    text_list.append(expected_file)
    # Then
    assert actual_file_names == text_list


correct_json = '''{
  "ReceiptInfo": {
    "merchant": "Minit",
    "address": "26 Kekela Street",
    "city": "Hilo",
    "state": "HI",
    "phoneNumber": "(xxx) xxx-xxxx",
    "tax": 0,
    "total": 23.75,
    "receiptDate": "07/08/2023",
    "receiptTime": "12:12:18",
    "ITEMS": [
      {
        "description": "Pump #02 - Self",
        "quantity": 1,
        "unitPrice": 5.209,
        "totalPrice": 23.75,
        "discountAmount": 0
      }
    ]
  }
}
'''

receipt_text = '''Minit <UNKNOWN> <UNKNOWN>
26 Kekela Street
Hilo
HI, 96720
7/8/2023
12:12:18
Pump #02 - Self <UNKNOWN>
Unleaded - R
5.209
Price/Gal
4.559
Fuel Ttl
$23.75
HFN <UNKNOWN> Fleet Price
Micah <UNKNOWN>
<UNKNOWN>
O <UNKNOWN>
07/08/2023 12:10 : 27
I agree to pay the
above Total Amount
according to card
Issuer <UNKNOWN>
TRAN : 1718674
HFN <UNKNOWN> Fleet <UNKNOWN>'''

CHATGPT_PROMPT = '''Please analyze the provided receipt and extract relevant information to fill in the following structured format:
{
  "ReceiptInfo": {
    "merchant": "(string value)",
    "address": "(string value)", (split into street address, city, and state)
    "city": "(string value)",
    "state": "(string value)",
    "phoneNumber": "(string value)",
    "tax": "(float value)", (in dollars)
    "total": "(float value)", (in dollars)
    "receiptDate": "(string value)",
    "receiptTime": "(string value)", (if available)
    "ITEMS": [
      {
        "description": "(string value)",
        "quantity": "(integer value)",
        "unitPrice": "(float value)",
        "totalPrice": "(float value)",
        "discountAmount": "(float value)" if any
      }, ...
    ]
  }
}
Remember to check for any discounts or special offers applied to the items and reflect these in the item details. Make sure to end the json object and make sure it's in json format.
1. tax, total, unitPrice, totalPrice, discountAmount in float value, and quantity in integer value
2. ignore all <UNKNOWN> in the text
3. Your response should start with { and end with },
4. make sure close all ReceiptInfo and use , to separate different ReceiptInfo

example: """Marley's Shop
123 Long Rd
Kailua, HI 67530
(808) 555-1234
CASHIER: JOHN
REGISTER #: 6
04/12/2023
Transaction ID: 5769009
PRICE   QTY  TOTAL
APPLES (1 lb)
2.99 2 5.98  1001
-1.00  999
Choco Dream Cookies
7.59 1 7.59   1001
SUBTOTAL
13.57
SALES TAX 8.5%
1.15
TOTAL
-14.72
VISA CARD            14.72
CARD#: **1234
REFERENCE#: 6789
THANK YOU FOR SHOPPING WITH US!
"""

from example should get:
{
  "ReceiptInfo": {
    "merchant": "Marley's Shop",
    "address": "123 Long Rd",
    "city": "Kailua",
    "state": "HI",
    "phoneNumber": "(xxx) xxx-xxxx",
    "tax": 1.15,
    "total": 14.72,
    "receiptDate": "04/12/2023",
    "receiptTime": "Transaction ID: 5769009",
    "ITEMS": [
      {
        "description": "APPLES (1 lb)",
        "quantity": 2,
        "unitPrice": 2.99,
        "totalPrice": 5.98,
        "discountAmount": 1.00
      },
      {
        "description": "Choco Dream Cookies",
        "quantity": 1,
        "unitPrice": 7.59,
        "totalPrice": 7.59,
        "discountAmount": 0
      }
    ]
  }
}
'''

def generate_receipt_json(receipt_text, apikey):
    '''
    Generates a receipt JSON given receipt text using ChatGPT.

    Parameters:
    receipt_text (str): The text to feed ChatGPT.

    Returns:
    dict or None: The receipt JSON as a dictionary or None if ChatGPT generates invalid JSON.
    '''

    llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0, openai_api_key=apikey, max_tokens=1056)
    response = llm(receipt_text)

    schema = {
        "type": "object",
        "properties": {
            "ReceiptInfo": {
                "type": "object",
                "properties": {
                    "merchant": {"type": "string"},
                    "address": {"type": "string"},
                    "city": {"type": "string"},
                    "state": {"type": "string"},
                    "phoneNumber": {"type": "string"},
                    "tax": {"type": "number"},
                    "total": {"type": "number"},
                    "receiptDate": {"type": "string"},
                    "ITEMS": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string"},
                                "quantity": {"type": "number"},
                                "unitPrice": {"type": "number"},
                                "totalPrice": {"type": "number"},
                                "discountAmount": {"type": "number"}
                            },
                        },
                    },
                },
            },
        },
    }

    return response
def test_process_and_validate_json(apikey):
    response = generate_receipt_json(CHATGPT_PROMPT + receipt_text, apikey)
    # Given
    schema = {
        "type": "object",
        "properties": {
            "ReceiptInfo": {
                "type": "object",
                "properties": {
                    "merchant": {"type": "string"},
                    "address": {"type": "string"},
                    "city": {"type": "string"},
                    "state": {"type": "string"},
                    "phoneNumber": {"type": "string"},
                    "tax": {"type": "number"},
                    "total": {"type": "number"},
                    "receiptDate": {"type": "string"},
                    "ITEMS": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string"},
                                "quantity": {"type": "number"},
                                "unitPrice": {"type": "number"},
                                "totalPrice": {"type": "number"},
                                "discountAmount": {"type": "number"}
                            },
                        },
                    },
                },
            },
        },
    }

    # When
    test_json = json.loads(process_and_validate_json(response, schema))

    test_correct_json = json.loads(correct_json)
    # Then
    assert test_json == test_correct_json

def test_receipt_texts_to_json_list():
    assert os.path.isfile('../processed_receipts/receipts.json')

def test_receipts_json_to_csv():
    assert os.path.isfile('../processed_receipts/vendors_and_products.csv')

#2
def test_generate_embeddings():
    word = "apple"
    embeddings = generate_embeddings(word)
    assert embeddings.shape == (1, 1024)

def test_convert_to_embeddings_df():
    df = pd.DataFrame({"fruit": ["apple", "banana", "cherry"]})
    embeddings_df = convert_to_embeddings_df(df)
    assert embeddings_df.shape == (3, 1024)

def test_create_embedded_vendor_database():
    assert os.path.isfile("../databases/vendor/embedding/embedded_vendor_database.csv")

def test_create_embedded_product_database():
    assert os.path.isfile("../databases/product/embedding/embedded_product_database.csv")


#3
def test_split_database():
    file_path = "../databases/vendor/embedding/embedded_vendor_database.csv"
    X, y = split_database(file_path)
    assert X.shape[1] == 1024
    assert y.shape[0] == X.shape[0]

def test_cosine_similarity_clf():
    X_train = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
    y_train = pd.Series(["a", "b", "c", "d", "e"])
    X_test = pd.DataFrame([[7, 8, 9]])
    y_pred = cosine_similarity_clf(X_train, y_train, X_test)
    assert y_pred == "b"

def test_get_vendor_category():
    assert os.path.isfile("../predictions/vendor_category_predictions.csv")

def test_process_list():
    row = pd.Series({"Products": ["apple", "banana"], "Vendors": "Walmart"})
    X_test, items = process_list(row)
    assert X_test == ["apple Walmart", "banana Walmart"]
    assert items == ["apple", "banana"]

def test_get_product_category():
    assert os.path.isfile("../predictions/product_category_predictions.csv")
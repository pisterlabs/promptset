import os
import time
import json
import pandas as pd
import numpy as np
import torch
import glob

from langchain.llms import OpenAI
from marshmallow import ValidationError
from transformers import BertModel, BertTokenizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from jsonschema import validate

UPDATE_RECEIPTS = False # Set this to true if we've added new reciept data in
UPDATE_VENDOR_DATABASE = False # Set this to true if we've added more categories/examples to the vendor database
UPDATE_PRODUCT_DATABASE = False # Set this to true if we've added more categories/examples to the product database


#Section 1:
RECEIPTS_INPUT = '../receipts/text'
RECEIPTS_OUTPUT = '../processed_receipts'

OPENAI_API_KEY = ''

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


def read_text_files(folder_path):
    '''
    Reads all text files within a folder path.

    Parameters:
    folder_path (str): The folder path.

    Returns:
    list[str]: The list of all file names contained at the folder path.
    '''

    text_list = []

    if not os.path.isdir(folder_path):
        print('Invalid folder path.')
        return None

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path) and filename.endswith('.txt'):
            with open(file_path, 'r') as file:
                file_content = file.read()
                text_list.append(file_content)  # Append file content as a string to the list

    return text_list


def process_and_validate_json(response, schema):
    '''
    Processes and validates a JSON string.

    Parameters:
    response (str): The folder path.
    schema (dict): The schema to validate against.

    Returns:
    dict or None: The JSON as a dictionary or None if invalid JSON.
    '''

    # Find the index of the first '{'
    brace_index = response.find('{')

    # If '{' is found and it's not the first character
    if brace_index != -1:
        # Extract JSON from the substring starting from the first '{'
        extracted_json = response[brace_index:]

        # Validate the extracted JSON against the provided schema
        try:
            validate(instance=json.loads(extracted_json), schema=schema)
            return extracted_json
        except json.JSONDecodeError as e:
            print(f'Error decoding JSON: {e}')
        except ValidationError as e:
            print(f'JSON validation error: {e}')

    # Return None if '{' is not found or it's the first character
    return None


def generate_receipt_json(receipt_text):
    '''
    Generates a receipt JSON given receipt text using ChatGPT.

    Parameters:
    receipt_text (str): The text to feed ChatGPT.

    Returns:
    dict or None: The receipt JSON as a dictionary or None if ChatGPT generates invalid JSON.
    '''

    llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0, openai_api_key=OPENAI_API_KEY, max_tokens=1056)
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

    return process_and_validate_json(response, schema)


def receipt_texts_to_json_list():
    '''
    Converts all receipt texts located at RECEIPTS_INPUT into a file of a list of JSONs named entities.json.
    '''

    output_path = RECEIPTS_OUTPUT + '/receipts.json'

    receipts = read_text_files(RECEIPTS_INPUT)

    receipts_json = []
    errorReceipts = []
    files_processed = 0
    for receipt in receipts:
        receipt_json = json.loads(generate_receipt_json(CHATGPT_PROMPT + receipt))
        receipts_json.append(receipt_json)
        files_processed += 1

    with open(output_path, 'w') as file:
        json.dump(receipts_json, file, indent=4)


def receipts_json_to_csv():
    '''
    Converts JSON list of receipts stored in entities.json into CSV of only vendor and product descriptions.
    '''

    # Read and parse the JSON file
    with open(RECEIPTS_OUTPUT + '/receipts.json', 'r') as file:
        data = json.load(file)

    entry_number = 0

    # Initialize lists to store data
    merchants = []
    descriptions = []

    # Iterate through the data
    for entry in data:
        entry_number += 1
        merchant = entry["ReceiptInfo"]["merchant"]
        items = entry["ReceiptInfo"]["ITEMS"]

        # Initialize a list to store cleaned descriptions for this entry
        cleaned_descriptions = []

        # Remove "number+space" occurrences in the descriptions and add to the list
        for item in items:
            description = item.get('description', 'No Description')
            cleaned_description = ' '.join(word for word in description.split() if not word.isdigit())
            cleaned_descriptions.append(cleaned_description)

        # Remove "UNKNOWN," "<UNKNOWN>," and "unknown" from the merchant field
        merchant = merchant.replace("UNKNOWN", "").replace("<UNKNOWN>", "").replace("unknown", "").replace("<>", "")

        # Add the merchant and descriptions to the respective lists
        merchants.append(merchant)
        descriptions.append(cleaned_descriptions)

    # Create a DataFrame and save as CSV
    entities_df = pd.DataFrame({
        'Vendors': merchants,
        'Products': descriptions
    })
    entities_df.to_csv(RECEIPTS_OUTPUT + '/vendors_and_products.csv', index=0)


#Section 2
if UPDATE_RECEIPTS:
    receipt_texts_to_json_list()
    receipts_json_to_csv()

model_name = "BAAI/bge-large-en"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)


def generate_embeddings(word):
    '''
    Generates a vector of embeddings given a word/sentence.

    Parameters:
    word (str): The word/sentence.

    Returns:
    tensor(1, 1024): The vector of embeddings.
    '''

    inputs = tokenizer(word, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling of token embeddings
    return embeddings


def convert_to_embeddings_df(df):
    '''
    Convert words in a DataFrame column to embeddings.

    Parameters:
    df (dataframe): The dataframe.

    Returns:
    dataframe: The dataframe of embeddings.
    '''

    embeddings = [generate_embeddings(x) for x in df.iloc[:, 0]]
    dfs = []
    for embedding in embeddings:
        dfs.append(pd.DataFrame(embedding))
    return pd.concat(dfs)


def create_embedded_vendor_database():
    '''
    Create vector of embeddings database from vendor word databases.
    Outputs to ./databases/vendor/embedding.
    '''

    vendor_database = pd.DataFrame()

    csv_files = glob.glob(os.path.join('.', 'databases', 'vendor', 'word', '*.csv'))
    for file in csv_files:
        category = os.path.split(file)[-1]
        category_name = category.replace('.csv', '').replace('_', ' ')

        new_category = pd.read_csv(file, encoding='latin-1')
        new_column = convert_to_embeddings_df(new_category)
        new_column['Category'] = category_name

        vendor_database = pd.concat([vendor_database, new_column], ignore_index=True, axis=0)
    vendor_database.to_csv("../databases/vendor/embedding/embedded_vendor_database.csv")

    return vendor_database


def create_embedded_product_database():
    '''
    Create vector of embeddings database from product word databases.
    Outputs to ./databases/product/embedding.
    '''

    product_database = pd.DataFrame()

    # Loop through subfolders of product CSV files
    for root, dirs, files in os.walk(os.path.join('.', 'databases', 'product', 'word')):
        for file in files:
            if file.endswith('.csv'):
                csv_file_path = os.path.join(root, file)  # Get the absolute path of the CSV file
                category = os.path.split(file)[-1]
                category_name = category.replace('.csv', '').replace('_', ' ')

                new_category = pd.read_csv(csv_file_path, encoding='latin-1')
                new_column = convert_to_embeddings_df(new_category)
                new_column['Category'] = category_name

                product_database = pd.concat([product_database, new_column], ignore_index=True, axis=0)
    product_database.to_csv('../databases/product/embedding/embedded_product_database.csv')

    return product_database


#Section 3
if UPDATE_VENDOR_DATABASE:
    create_embedded_vendor_database()

if UPDATE_PRODUCT_DATABASE:
    create_embedded_product_database()


def split_database(file_path):
    '''
    Splits vector database csv into X and y.

    Parameters:
    file_path (str): The path to the vector database.

    Returns:
    tuple(dataframe, dataframe): X and y.
    '''

    df = pd.read_csv(file_path)
    df = df.drop('Unnamed: 0', axis=1)

    # Creating variables from database values
    X = df.drop('Category', axis=1)
    y = df['Category']

    return X, y

def cosine_similarity_clf(X_train, y_train, X_test):
    df = (X_train.values @ X_test.values.T) / np.linalg.norm(X_train.values, axis=1).reshape(-1, 1) / np.linalg.norm(X_test.values, axis=1)
    return y_train[pd.DataFrame(df, columns=X_test.index, index=X_train.index).idxmax()[0]]


def get_vendor_category():
    '''
    Runs classification of vendor category on all receipts.
    Outputs prediction results to ./predictions/vendor_category_predictions.csv.
    '''

    X_train, y_train = split_database('../databases/vendor/embedding/embedded_vendor_database.csv')

    receipts = pd.read_csv("../processed_receipts/vendors_and_products.csv")
    vendors = receipts['Vendors'].to_frame()

    vendors_embeddings = convert_to_embeddings_df(vendors)
    X_test = vendors_embeddings

    results = pd.DataFrame({'Prediction': cosine_similarity_clf(X_train, y_train, X_test)}).reset_index(drop=True)
    result_df = pd.concat([vendors, results], axis=1)
    return result_df


# Dump predictions to csv
get_vendor_category().to_csv('../predictions/vendor_category_predictions.csv')


def process_list(row):
    '''
    Helper function to add vendor to product description to improve classifcation performance.

    Parameters:
    row (dataframe): The row of a dataframe.

    Returns:
    tuple(list[str], list[str]): The X_test of vendor and product description combined
                                 and the product decscription themselves.
    '''

    X_test, items = [], []
    for item in row['Products']:
        X_test.append(item + " " + row['Vendors'])
        items.append(item)
    return X_test, items


def get_product_category():
    '''
    Runs classification of product category on all receipts.
    Outputs prediction results to ./predictions/product_category_predictions.csv.
    '''

    X_train, y_train = split_database('../databases/product/embedding/embedded_product_database.csv')

    receipts = pd.read_csv('../processed_receipts/vendors_and_products.csv')
    receipts['Products'] = receipts['Products'].apply(eval)
    receipts = receipts.apply(process_list, axis=1)

    X_test = [item[0] for item in receipts]
    items = [item[1] for item in receipts]

    receipt_items, merchant_items = [], []
    for i, product in enumerate(items):
        product = items[i]
        for item in product:
            receipt_items.append(item)
        for merchant_item in X_test[i]:
            merchant_items.append(merchant_item)

    X_test = pd.DataFrame(merchant_items)
    receipt_embeddings = convert_to_embeddings_df(X_test)
    X_test = receipt_embeddings

    results = pd.DataFrame({'Prediction': cosine_similarity_clf(X_train, y_train, X_test)}).reset_index(drop=True)
    receipt_items = pd.DataFrame(receipt_items)
    result_df = pd.concat([receipt_items, results], axis=1)

    return result_df


# Dump predictions to csv
get_product_category().to_csv('../predictions/product_category_predictions.csv')
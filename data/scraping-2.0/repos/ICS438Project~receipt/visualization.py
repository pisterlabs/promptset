import os
import glob
import json
import torch
import chardet
import zipfile
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

from langchain.llms import OpenAI
from transformers import BertModel, BertTokenizer
from sklearn.neighbors import KNeighborsClassifier
from jsonschema import validate

#
# Constants
#

DATA_PATH = './'
CHATGPT_PROMPT = \
    '''Please analyze the provided receipt and extract relevant information to fill in the following structured format:
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

#
# Receipt Parsing Helper Functions
#

def validate_json(entities):
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

    return validate(instance=json.loads(entities), schema=schema)

def ensure_starts_with_brace(response):
    # Find the index of the first '{'
    brace_index = response.find('{')

    # If '{' is found and it's not the first character
    if brace_index != -1:
        # Return the substring starting from the first '{'
        return response[brace_index:]

    # Return the original response if '{' is not found
    return response

def generate_response(input_text, openai_api_key):
    llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0, openai_api_key=openai_api_key, max_tokens=1056)
    response = llm(input_text)
    response = ensure_starts_with_brace(response)
    validate_json(response)
    return response

def cosine_similarity_clf(X_train, y_train, X_test):
    df = (X_train.values @ X_test.values.T) / np.linalg.norm(X_train.values, axis=1).reshape(-1, 1) / np.linalg.norm(X_test.values, axis=1)
    return y_train[pd.DataFrame(df, columns=X_test.index, index=X_train.index).idxmax()[0]]

def get_embedded_database(zip_file_path, csv_file_name):
    # Check if the zip file exists
    if not os.path.isfile(zip_file_path):
        print(f"File not found: {zip_file_path}")
        return None, None

    # Extract the CSV file from the ZIP archive
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Extract all the contents into the directory of the ZIP file
        zip_ref.extractall(os.path.dirname(zip_file_path))

    # Path to the extracted CSV file
    extracted_csv_path = os.path.join(os.path.dirname(zip_file_path), csv_file_name)

    # Check if the extracted CSV file exists
    if not os.path.isfile(extracted_csv_path):
        print(f"Extracted file not found: {extracted_csv_path}")
        return None, None

    # Read the CSV file
    df = pd.read_csv(zip_file_path)
    df = df.drop('Unnamed: 0', axis=1)

    # Creating variables from database values
    X = df.drop('Category', axis=1)
    y = df['Category']

    return X, y

def get_vendor_category(vendor):
    X_train, y_train = get_embedded_database(f'{DATA_PATH}databases/vendor/embedding/embedded_vendor_database.zip',
                                           "embedded_vendor_database.csv")

    # Convert the vendor into dataframe
    merchants_df = pd.DataFrame({'Vendor': [vendor]})
    merchants_embeddings = convert_to_embeddings_df(merchants_df)
    X_test = merchants_embeddings

    # Run the prediction model
    results = pd.DataFrame({'Prediction': [cosine_similarity_clf(X_train, y_train, X_test)]}).reset_index(drop=True)
    result_df = pd.concat([merchants_df, results], axis=1)
    return result_df

def get_product_category(products):
    # Load the embedded product database
    X_train, y_train = get_embedded_database(f'{DATA_PATH}databases/product/embedding/embedded_product_database.zip',
                                           "embedded_product_database.csv")

    # Convert descriptions to DataFrame
    products_df = pd.DataFrame({'Products': products})
    products_embeddings = convert_to_embeddings_df(products_df)
    X_test =  products_embeddings

    # Run the prediction model
    results = pd.DataFrame({'Prediction': cosine_similarity_clf(X_train, y_train, X_test)}).reset_index(drop=True)
    result_df = pd.concat([products_df, results], axis=1)
    return result_df

def generate_embeddings(word):
    inputs = tokenizer(word, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling of token embeddings
    return embeddings

# Function to convert words in a DataFrame column to embeddings
def convert_to_embeddings_df(df):
    embeddings = [generate_embeddings(x) for x in df.iloc[:, 0]]
    dfs = []
    for embedding in embeddings:
        dfs.append(pd.DataFrame(embedding))
    return pd.concat(dfs)

#
# Streamlit Components
#

def navbar():
    st.title("Receipt Parsing and Analytics System Dashboard")
    st.sidebar.title("Navigator")

    # Initialize a session state variable to track the active dashboard
    if 'active_dashboard' not in st.session_state:
        st.session_state['active_dashboard'] = 'Receipt Parsing'
    if st.sidebar.button('Receipt Parsing'):
        st.session_state['active_dashboard'] = 'Receipt Parsing'
    if st.sidebar.button('Receipt Dashboard'):
        st.session_state['active_dashboard'] = 'Receipt'
    if st.sidebar.button('Database Dashboard'):
        st.session_state['active_dashboard'] = 'Database'

def receipt_parsing():
    if st.session_state['active_dashboard'] == 'Receipt Parsing':
        st.title("Receipt Parsing")

        st.subheader("Step 1: Get your own OpenAI API key")
        openai_api_key = st.text_input("Enter your OpenAI API key", type='password')
        if not openai_api_key.startswith('sk-'):
            st.warning('Please enter your OpenAI API key!', icon='⚠')

        st.subheader("Step 2: Input Receipt Text")
        uploaded_receipt = st.file_uploader("Upload receipts", type=['txt'], accept_multiple_files=False)

        receipt = ''
        if uploaded_receipt:
            receipt = uploaded_receipt.read().decode('utf-8')
            st.text_area("Receipt Content", receipt, height=500)
        submitted = st.button('Submit')

        receipt_json = {}
        if submitted and openai_api_key.startswith('sk-'):
            try:
                receipt_json = json.loads(generate_response(CHATGPT_PROMPT + receipt, openai_api_key))
                st.success("Receipt has been processed!")
            except json.JSONDecodeError as e:
                st.error(f"JSON Decode Error for receipt: {e}")

        st.subheader("Step 3: Run Predictions")

        if receipt_json and isinstance(receipt_json, dict):
            # Extract merchant and items from receipt_json
            vendor = receipt_json['ReceiptInfo']['merchant']
            products = receipt_json['ReceiptInfo']['ITEMS']

            # Get predictions
            vendor_prediction = get_vendor_category(vendor)

            descriptions = [product['description'] for product in products]
            product_predictions = get_product_category(descriptions)

            # Display Vendor Category Prediction
            st.subheader("Vendor Category Prediction")
            if not vendor_prediction.empty:
                st.write(vendor_prediction)

            # Display Product Category Predictions
            st.subheader("Product Category Predictions")
            if not product_predictions.empty:
                st.write(product_predictions)
        else:
            st.write("No receipt data available for prediction.")


def receipt_dashboard():
    # Display content based on the active dashboard
    if st.session_state['active_dashboard'] == 'Receipt':
        st.title("Receipt Dashboard")
        st.subheader("Vender Database (Receipts)")

        # Path to the vendor database directory
        vendor_db_path = f'{DATA_PATH}predictions/vendor_category_predictions.csv'  # Update with the actual path
        df = pd.read_csv(vendor_db_path)

        # Count the frequency of unique values in the 'Prediction' column
        pred_counts = df['Prediction'].value_counts()

        # Create a bar chart using Plotly
        fig = px.bar(pred_counts, x=pred_counts.index, y=pred_counts.values,
                     labels={'x': 'Prediction', 'y': 'Number of Data'},
                     title='Distribution of Predictions(Vendor Category)')

        fig.update_layout(
            autosize=False,
            width=1000,
            height=600,
            title_font_size=24,
            font=dict(size=18),
            xaxis_title_font_size=20,
            yaxis_title_font_size=20,
            xaxis_tickfont_size=16,
            yaxis_tickfont_size=16
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.subheader("Product Database (Receipts)")
        product_db_path = f"{DATA_PATH}predictions/product_category_predictions.csv"  # Update with actual path

        df = pd.read_csv(product_db_path)

        # Count the frequency of unique values in the 'Prediction' column
        pred_counts = df['Prediction'].value_counts()

        # Create a bar chart using Plotly
        fig = px.bar(pred_counts, x=pred_counts.index, y=pred_counts.values,
                     labels={'x': 'Prediction', 'y': 'Number of Data'},
                     title='Distribution of Predictions(Product Category)')

        fig.update_layout(
            xaxis_tickangle=45,
            autosize=False,
            width=1000,
            height=600,
            title_font_size=24,
            font=dict(size=18),
            xaxis_title_font_size=20,
            yaxis_title_font_size=20,
            xaxis_tickfont_size=16,
            yaxis_tickfont_size=16
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def database_dashboard():
    if st.session_state['active_dashboard'] == 'Database':
        st.title("Database Dashboard")

        selected_database = st.selectbox(
            "Select Database",
            ["Select a Database", "Vendor Database", "Product Database"]
        )

        if selected_database == "Vendor Database":

            # Function to count rows in a CSV file
            def count_rows(csv_file_path):
                df = pd.read_csv(csv_file_path)
                return len(df)

            st.subheader("Vender Database")

            # Path to the vendor database directory
            vendor_db_path = f"{DATA_PATH}databases/vendor/word"  # Update with the actual path

            # List CSV files in the vendor database directory
            vendor_csv_files = [file for file in os.listdir(vendor_db_path) if file.endswith('.csv')]

            # Count rows in each CSV file
            vendor_row_counts = {file.replace('.csv', ''): count_rows(os.path.join(vendor_db_path, file)) for file in
                                 vendor_csv_files}

            # Prepare data for plotting
            vendor_names = list(vendor_row_counts.keys())
            row_counts = list(vendor_row_counts.values())

            # Create a bar chart using Plotly
            fig = px.bar(x=vendor_names, y=row_counts,
                         labels={'x': 'Vendor Category', 'y': 'Row Count'},
                         title='Total Number of Data Vendor Category')

            # Customize plot size and font
            fig.update_layout(
                autosize=False,
                width=1000,
                height=600,
                title_font_size=24,
                font=dict(size=18),
                xaxis_title_font_size=20,
                yaxis_title_font_size=20,
                xaxis_tickfont_size=16,
                yaxis_tickfont_size=16
            )

            # Display the plot in Streamlit
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        if selected_database == "Product Database":
            # Function to count rows in CSV files of a folder
            def find_encoding(file_path):
                with open(file_path, 'rb') as file:
                    result = chardet.detect(file.read())
                    return result['encoding']

            # Modified function to count rows in CSV files of a folder
            def count_rows_in_folder(folder_path):
                total_rows = 0
                for file in os.listdir(folder_path):
                    if file.endswith('.csv'):
                        file_path = os.path.join(folder_path, file)
                        encoding = find_encoding(file_path)
                        df = pd.read_csv(file_path, encoding=encoding)
                        total_rows += len(df)
                return total_rows

            # Function to get row counts for each CSV in a folder
            def row_counts_per_csv(folder_path):
                counts = {}
                for file in os.listdir(folder_path):
                    if file.endswith('.csv'):
                        try:
                            # Try reading with utf-8 encoding
                            df = pd.read_csv(os.path.join(folder_path, file), encoding='utf-8')
                        except UnicodeDecodeError:
                            # If utf-8 fails, try a different encoding
                            df = pd.read_csv(os.path.join(folder_path, file), encoding='ISO-8859-1')
                        counts[file] = len(df)
                return counts

            # Product Database
            st.subheader("Product Database")
            product_db_path = f"{DATA_PATH}databases/product/word"  # Update with actual path

            # Plot total row count per folder
            folder_row_counts = {folder: count_rows_in_folder(os.path.join(product_db_path, folder)) for folder in
                                 os.listdir(product_db_path)}
            folders = list(folder_row_counts.keys())
            row_counts = list(folder_row_counts.values())

            # Create a bar chart
            fig = px.bar(x=folders, y=row_counts,
                         labels={'x': 'Product Category', 'y': 'Number of Data'},
                         title='Total Number of Data of all main product categories')

            # Customize the hover data
            fig.update_traces(hoverinfo='y+name')

            # Update layout for larger figure and increased font sizes
            fig.update_layout(
                xaxis_tickangle=45,
                autosize=False,
                width=1000,  # Width of the figure in pixels
                height=600,  # Height of the figure in pixels
                title_font_size=24,  # Title font size
                font=dict(size=18),  # General font size for axis labels, etc.
                xaxis_title_font_size=20,  # X-axis title font size
                yaxis_title_font_size=20,  # Y-axis title font size
                xaxis_tickfont_size=16,  # X-axis tick font size
                yaxis_tickfont_size=16  # Y-axis tick font size
            )

            # Display the plot in Streamlit
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # SelectBox for selecting a folder
            selected_folder = st.selectbox("Select a Product Category", os.listdir(product_db_path))

            # Get row counts for each CSV in the selected folder
            csv_row_counts = row_counts_per_csv(os.path.join(product_db_path, selected_folder))

            labels = [filename.replace('.csv', '') for filename in csv_row_counts.keys()]
            values = csv_row_counts.values()

            # Create a pie chart
            fig = px.pie(names=labels, values=values, title=f'Percentage of Data in {selected_folder}',
                         labels={'value': 'Number of Data'})
            # Update layout for larger figure and increased font sizes
            fig.update_layout(
                autosize=False,
                width=800,  # Width of the figure in pixels
                height=600,  # Height of the figure in pixels
                title_font_size=24,  # Title font size
                font=dict(size=18),  # General font size for axis labels, etc.
                xaxis_title_font_size=20,  # X-axis title font size
                yaxis_title_font_size=20,  # Y-axis title font size
                xaxis_tickfont_size=16,  # X-axis tick font size
                yaxis_tickfont_size=16  # Y-axis tick font size
            )

            # Display the plot in Streamlit
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

if __name__ == '__main__':
    # Load the BertTokenizer and BertModel
    model_name = "BAAI/bge-large-en"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    # Load Streamlit components
    navbar()
    receipt_parsing()
    receipt_dashboard()
    database_dashboard()
    
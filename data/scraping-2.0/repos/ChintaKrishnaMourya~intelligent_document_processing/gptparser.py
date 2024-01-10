import openai
import json
import streamlit as st
import pandas as pd
import os
import time

api_key = st.secrets["api_key"]
openai.api_key = api_key

def gpt_parser(text):
    prompt='''You are given a English text that is extrcted by using an OCR Tool. The extracted text may sometimes irregular or unclear. So try to understand them interms of bill of lading.
  Now, your task is to give right values for the given keys.
  Keys : 
  - "Invoice_Number"
  - "Invoice_Date"
  - "Due_Date"
  - "Bill_of_lading_number"
  - "Supplier_Information" (with sub-keys: "Supplier_Name," "Supplier_Address," "Supplier_Contact")
  - "Customer_Information" (with sub-keys: "Customer_Name," "Customer_Address," "Customer_Contact")
  - "Billing_Details" (with sub-keys for each line item: "Description," "Quantity," "Unit_Price," "Total_Price")
  - "Currency"
  - "Additional_Notes_or_Terms".
  If no right value found for above keys, give value as "None"
  Stricltly, give final output in JSON format. '''
    # Record the start time
    start_time = time.time()
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k", 
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": text}]
    )["choices"][0]["message"]["content"]
    
    # Parse the JSON response
    try:
        output_json = json.loads(completion)
    except json.JSONDecodeError as e:
        st.error(f"Error parsing JSON: {e}")
        return None

    df = pd.DataFrame.from_dict(output_json, orient='index') # Load the JSON data into a DataFrame
    df = df.T # Transpose the DataFrame to have columns as rows
    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    st.write(f"Time Taken for Parsing : {elapsed_time} seconds")
    return df

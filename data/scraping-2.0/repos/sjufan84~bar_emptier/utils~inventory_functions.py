import os
import streamlit as st
import pandas as pd
import openai
from dotenv import load_dotenv
from typing import Optional
load_dotenv()

# Get the OpenAI API key and org key from the .env file
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG")

if "df" not in st.session_state:
    st.session_state.df = None


class InventoryService:
    def __init__(self, inventory: Optional[dict] = None):
        # If there is an inventory passed, save it to the inventory attribute
        if inventory:
            self.inventory = inventory
        else:
            self.inventory = None
        
    
       
        
    # Define the function to process the file
    def process_and_format_file(self, uploaded_file):
        if uploaded_file == None:
            df = pd.read_csv('./resources/inventory.csv')
        elif uploaded_file.name.lower().endswith('csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.lower().endswith(('xlsx', 'xls')):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            raise ValueError('File must be a CSV or Excel file')

            
        # Convert the columns to match desired format
        df.columns = ['Name','Quantity','Volume per Unit (ml)','Cost per Unit']
        # Use the formatted dataframe to calculate the cost per ml, cost per oz, 
        # total amount (ml), total amount (oz), and total value and add these columns
        # Divide the total cost by the total amount to get the cost per ml
        df['Cost per ml'] = df['Cost per Unit'] / df['Volume per Unit (ml)']
        # Convert the cost per ml to cost per oz
        df['Cost per oz'] = round((df['Cost per ml'] / 0.033814), 3)
        # Calculate the total value of the inventory
        df['Total Amount (ml)'] = df['Quantity'] * df['Volume per Unit (ml)']
        df['Total Value'] = df['Total Amount (ml)'] * df['Cost per ml']

        st.session_state.df = df

        # Save the inventory to redis
        self.inventory = df.to_dict()

        # Return the dictionary
        return df

                
      

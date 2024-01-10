import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import website.mongo as mongo
import datetime
import openai
from dotenv import load_dotenv, find_dotenv
import os


# Get data from MongoDB
def create_dataframe(username):
    df = pd.DataFrame(columns=["Username","Date", "Product", "Expense"])
    receipts = mongo.get_receipts(username)
    for receipt in receipts:
        username = receipt["username"]
        date = receipt["date"]
        products = receipt["products"]
        for product in products:
            product_name = product
            product_price = products[product]

            df.loc[len(df.index)] = [username, date, product_name, product_price]
    
    def categorizing_products(df) :
        df['Category'] = ""
        product_list = df['Product'].tolist()
        string_list = str(product_list)
        

        # Set up the OpenAI API
        load_dotenv(find_dotenv())
        api_key = os.environ.get("API_KEY")

        openai.api_key = api_key

        # Call the API
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "user", 
                "content": f"Given the following list of items, kindly write what category should they fall under out of the following options: Household Essentials, Food and Beverage, Shopping, Entertainment, Miscellaneous. Give your answer in the following way: [product 1 category, product 2 category...]. Do not include anything else in your response. \n The list of items is: {string_list}"}
                ]
        )

        gpt_response = response['choices'][0]['message']['content']
        
        gpt_response = gpt_response.strip("[").strip("]")

        categories_list = gpt_response.split(", ")

        df['Category'] = ''
        df['Category'] = categories_list
        
        return df
    
    return categorizing_products(df)

#print(create_dataframe("yuvbindal"))



#df = categorizing_products(create_dataframe("yuvbindal"))

def get_category_totals(df):
    category_totals = df.groupby('Category')['Expense'].sum().to_dict()
    return category_totals

#print("Category Totals:", get_category_totals(df))

def get_category_percentages(df):
    percentages = (float(df.groupby('Category')['Expense']).sum() / float(df['Expense']).sum() * 100).round(1).to_dict()
    return percentages

#print("Category Percentages:", get_category_percentages(df))


#create a pie chart of the categories
def get_category_pie(df):
    category_totals = get_category_totals(df)
    category_percentages = get_category_percentages(df)
    labels = list(category_totals.keys())
    sizes = list(category_totals.values())
    explode = [0.1]*len(labels)
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.axis('equal')
    plt.title("Category Breakdown")
    fig1.savefig('static/images/category_pie.png')

#get_category_pie(df)

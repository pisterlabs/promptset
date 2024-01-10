from openai import OpenAI
import pandas as pd
from preprocess import preprocess
import sys
CHATBOT_CONTEXT = "You are to return python code ONLY (nothing else) - preferably a single line when possible. Your code should be functions that are run on a dataframe called transactions_df wih columns: "


def query(input_string, client):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": CHATBOT_CONTEXT},
            {"role": "user", "content": input_string}
        ]
    )
    return completion.choices[0].message.content 

def call(input_text):
    CHATBOT_CONTEXT = "You are to return python code ONLY (nothing else) - preferably a single line when possible. Your code should be functions that are run on a dataframe called transactions_df wih columns: "
    input_string = sys.argv[1]
    transactions_df = pd.read_csv("user_data.csv")
    transactions_df = transactions_df.rename(columns= {"amount":"transaction_amount"})
    transactions_df = preprocess(transactions_df)
    CHATBOT_CONTEXT += str(transactions_df.columns)
    spend_limits = {
        'Eating out': 300,
        'Coffee': 50,
        'Healthcare and Wellbeing': 150,
        'Travel': 100,
        'Retail shops': 250,
        'Supermarkets': 400,
    #     'Insurance and finance': 1000,
        'Bills': 200,
        'Transfer to accounts':500 
    }
    CHATBOT_CONTEXT+= ". We also have spending limits for each transaction category given in the following dictionary called spend_limits: " + str(spend_limits)
    print(CHATBOT_CONTEXT)
    client = OpenAI()
    res = "No queries yet"
    while True:
        input_string = input()
        input_string = "Return code ONLY (nothing else) that would answer the following query based on the transactions_df data: " + input_string + ". Store this in a variable called res. If you define a function, also give the code to run the function."
        command = query(input_string, client)
        exec(command)
        print(command)
        print(res)



if __name__ == "__main__":
    input_string = sys.argv[1]
    transactions_df = pd.read_csv("user_data.csv")
    transactions_df = transactions_df.rename(columns= {"amount":"transaction_amount"})
    transactions_df = preprocess(transactions_df)
    CHATBOT_CONTEXT += str(transactions_df.columns)
    spend_limits = {
        'Eating out': 300,
        'Coffee': 50,
        'Healthcare and Wellbeing': 150,
        'Travel': 100,
        'Retail shops': 250,
        'Supermarkets': 400,
    #     'Insurance and finance': 1000,
        'Bills': 200,
        'Transfer to accounts':500 
    }
    CHATBOT_CONTEXT+= ". We also have spending limits for each transaction category given in the following dictionary called spend_limits: " + str(spend_limits)
    print(CHATBOT_CONTEXT)
    client = OpenAI()
    res = "No queries yet"
    while True:
        input_string = input()
        input_string = "Return code ONLY (nothing else) that would answer the following query based on the transactions_df data: " + input_string + ". Store this in a variable called res. If you define a function, also give the code to run the function."
        command = query(input_string, client)
        exec(command)
        print(command)
        print(res)

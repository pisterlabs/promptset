import openai
import pandas as pd
import pandas as pd
from io import StringIO
import os
from dotenv import load_dotenv
load_dotenv()


# Get the OpenAI API key from the environment
api_key = os.getenv('OPENAI_API_KEY')

# Initialize the OpenAI API client
openai.api_key = api_key


import random

def read_csv_header(df):
    print("======")
    print(df.head())
    print("======")
    df_no =  df[df['loan_status']=='Rejected']
    df_yes =  df[df['loan_status']=='Approved']
    print(df_yes.head(1))
    header = pd.concat([df_yes.head(1),df_no.head(5)])
    print(header)
    header = df.head()
    columns = df.columns
    description = df.describe()
    # print("Describe", description)
    generated_data = generate_model_data([header , columns , description])
    print(generated_data,type(generated_data))
    lines = generated_data.split("\n")
    generated_data = "\n".join(lines[:-1])
    print("-------------------------------------------")
    print(generated_data)
    synthetic_data = pd.read_csv(StringIO(generated_data))
    # synthetic_data.to_csv("synthetic_data.csv")
    
    return synthetic_data

def generate_model_data(csv_data):
    prompt = f""" Generate 100 rows of data for the csv header where columns is : {', '.join(csv_data[0])} strictly as csv output without any missing values .Don't generate code.
                    in Loan_status column only "Rejected" value should come.\n
                    """
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=3000,  # Adjust as needed
        n=1,  # Number of responses to generate
        stop=None,  # You can specify a stop sequence to end the response
    )

    return response.choices[0].text.strip()
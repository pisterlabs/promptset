import pandas as pd
import openai
import ast

#! change before use
api_list = ['org-', 'sk-']

# Read the CSV file into a DataFrame
df = pd.read_csv('nasdaq_screener_1697135861885.csv')


def answer_questions(task, input, system):
    openai.organization = api_list[0]
    openai.api_key = api_list[1]
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[{'role': 'system', 'content': system},
                  {'role': 'user', 'content': f'task = ({task}), input = ({input})'}]
    )
    response_var = completion["choices"][0]["message"]["content"]
    return response_var


def classify_industry(sector, industry, sectors, industries):
    try:
        # Use the provided sectors and industries dictionaries
        task = "Your task is to classify input sector and industry using sectors and industries dictionaries"
        input = f"sectors = {sectors}, industries = {industries}, input sector = '{sector}', input industry = '{industry}'"
        system = "Return only a list of predicted values like [1, 4] first value for sector and the second for industry"
        response = answer_questions(task, input, system)

        # Parse the response as a list of integers
        predicted_values = ast.literal_eval(response)
        return predicted_values
    except Exception as e:
        print(f"An error in classify_industry() occurred: {e}")
        return [0, 0]


def get_company_general_info(symbol, sectors, industries):
    """Function to get company general information like cap, sector etc by calling with symbol argument"""
    result = df[df['Symbol'] == symbol]
    temp_result = []
    if not result.empty:
        for index, row in result.iterrows():
            temp = classify_industry(row['Sector'], row['Industry'], sectors, industries)
            temp_result = [row['Symbol'], row['Market Cap'], temp[0], temp[1], row['Country']]
        return temp_result
    else:
        print(f"No entries found for '{symbol}' in the 'Symbol' column.")

#! copy to use get_company_general_info() as imported function
sectors = {}
industries = {}
sectors_df = pd.read_csv('com_names\\sectors.csv')
industries_df = pd.read_csv('com_names\\industries.csv')
sectors = {row['name']: row['id'] for index, row in sectors_df.iterrows()}
industries = {row['name']: row['id'] for index, row in industries_df.iterrows()}

temp = get_company_general_info("AAC", sectors, industries)
print(temp)

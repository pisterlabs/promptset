from dotenv import load_dotenv,find_dotenv
from langchain.llms import OpenAI
from langchain.agents import create_csv_agent
import pandas as pd

xlsx_file = '../data/cleaned_dataAI.xlsx'
output_file = '../data/cleaned_dataAI.csv'

# Load environment variables
load_dotenv(find_dotenv())

data = pd.read_excel(xlsx_file, engine='openpyxl')
data = data.rename(columns={'CO2_production': 'CO2_emissions'})


data.to_csv(output_file, index=False)

agent = create_csv_agent(OpenAI(model_name="gpt-3.5-turbo",temperature=0.5), '../data/cleaned_dataAI.csv', verbose=True)

# Start an interactive loop
while True:
    # Get user input
    user_input = input("You: ")

    # Exit the loop if the user types 'exit'
    if user_input.lower() == 'exit':
        break

    # Get the agent's response
    response = agent.run(user_input)

    # Print the agent's response
    print("Agent:", response)
# --------------------------------------------------------------
# Import libraries
# --------------------------------------------------------------
import os 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from langchain.agents import create_pandas_dataframe_agent 
from langchain.llms import OpenAI 
from langchain.agents import create_pandas_dataframe_agent
import pandas as pd



llm = OpenAI(model="text-davinci-003", temperature=0,openai_api_key='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')

excel_path = "testDir\sampleincomestatement.xls"
xls = pd.ExcelFile(excel_path)

# Initialize the list to store tables
tables = []
# Loop through each sheet in the Excel file
for sheet_name in xls.sheet_names:
    # Read the sheet into a DataFrame
    df = pd.read_excel(xls, sheet_name)
    
    # Remove rows with null values
    df = df.dropna()
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Append the processed DataFrame to the tables list
    tables.append(df)

# --------------------------------------------------------------
# Initialize pandas dataframe agent
# --------------------------------------------------------------

agent = create_pandas_dataframe_agent(llm, tables, verbose=True)



agent.run("Give me a table with the assets and accounts receivable for every year.")

# agent.run("What is the Assets for 2022FY")

# agent.run("Has accounts receivable increased every year?")

# agent.run("In a table show me everything that has increased every year?")

# agent.run("Based on the data in the file what can you tell me")

# --------------------------------------------------------------
# Perform multiple-steps data exploration
# --------------------------------------------------------------

# agent.run("which are the top 5 jobs that have the highest median salary?")

# agent.run("what is the percentage of data scientists who are working full time?")

# agent.run("which company location has the most employees working remotely?")

# agent.run("what is the most frequent job position for senior-level employees?")

# agent.run(
#     "what are the categories of company size? What is the proportion of employees they have? What is the total salary they pay for their employees?"
# )
# agent.run(
#     "get median salaries of senior-level data scientists for each company size and plot them in a bar plot."
# )


# # --------------------------------------------------------------
# # Perform basic & multiple-steps data exploration for both dataframes
# # --------------------------------------------------------------

# agent.run("how many rows and columns are there for each dataframe?")

# agent.run(
#     "what are the differences in median salary for data scientists among the dataframes?"
# )
# agent.run(
#     "how many people were hired for each of the dataframe? what are the percentages of experience levels?"
# )
# agent.run(
#     "what is the median salary of senior data scientists for df2, given there is a 10% increment?"
# )
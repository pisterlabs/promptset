from dotenv import load_dotenv
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from Modules.LLM import getKUKA_LLM

#Import API Keys
load_dotenv()

import langchain.schema
from langchain.agents import create_pandas_dataframe_agent

# chat completion llm
llm = getKUKA_LLM()

#LoadXLSX() Data in Pandas Dataframe
def LoadExcel():
    """
    Loads an Excel file and creates a pandas DataFrame agent.

    Returns:
        agent (pandas DataFrame): The pandas DataFrame agent created from the loaded Excel file.
    """
    # Open file dialog to select an xlsx file
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx")])

    # Read the xlsx file into a pandas DataFrame
    df = pd.read_excel(file_path)

    chat = llm
    agent = create_pandas_dataframe_agent(chat, df, verbose=True)

    return agent

def RunExcelQuery():
    """
    Runs an Excel query using the agent loaded from LoadExcel().
    The user can enter queries until they type 'exit' or 'quit'.
    Prints the answer obtained from the agent.

    Args:
        None

    Returns:
        None
    """
    agent = LoadExcel()
    while True:
        query = input("Enter your query: ")
        if query.lower() == "exit" or query.lower() == "quit":
            break
        try:
            agent.run(str(query))
        except langchain.schema.OutputParserException as e:
            # Extract the message from the exception
            message = str(e)
            # The message is in the form "Could not parse LLM output: `...`"
            # So, we can split it by the backticks and take the second element
            answer = message.split('`')[1]
            print("\n\nAnswer: ", answer)

if __name__ == "__main__":

    RunExcelQuery()
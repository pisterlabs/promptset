'''
To run this streamlit app, run the following command in your terminal:
streamlit run src/pandas_df_agent_p2_app.py

Input your question in the text box and click 'Ask' to get an answer.

Sample Questions for input
Q1: PALM API: "Find me the top 10 places in terms of 'E.coli count (CFU/g)"

Q2: PALM API: "Find me the places with the 10 lowest 'E.coli count (CFU/g)'"

Q3: PALM API:"Find me the top 10 cheapest places"

Q4: PALM API:"Find me the top 10 cheapest places, in terms of "Cost per gram"

Q5: PALM API: 
Draw a scatter plot showing relationship between E.coli count (CFU/g) and cost.

Q6: PALM API: 
Discard outliers in E.coli and then Draw a scatter plot showing relationship 
between E.coli count (CFU/g) and cost

etc


'''
# Importing libraries
import os
import sys
import pandas as pd
import streamlit as st
from io import StringIO
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import VertexAI
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import seaborn as sns
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './aiap-13-ds-7e16bb946970.json'
# Initialize the model
model = VertexAI(temperature=0)


def load_data(files):
    dataframes = []  # List to store individual dataframes from each file

    for file in files:
        # Extracting the batch number from the file name
        # batch_number = file.name.split('_')[2]
        df = pd.read_csv(file)  # Read the CSV file into a dataframe
        # Create a 'Batch' column and assign the batch number
        # df['Batch'] = batch_number
        dataframes.append(df)  # Append the dataframe to the list

    # Concatenate all dataframes into a single dataframe
    combined_df = pd.concat(dataframes, ignore_index=True)

    combined_df.rename(columns={
        'E.coli count (CFU/g)\n(>490000 is replaced by 490001, <10 is replaced by 9 for sorting)': 'E.coli count (CFU/g)'}, inplace=True)
    combined_df.rename(
        columns={'Rating (10/10) (Angel\'s rating)': 'Rating'}, inplace=True)
    combined_df.rename(
        columns={'Stall (Review >10 as of 2023 March 21)': 'Stall'}, inplace=True)

    combined_df.dropna(inplace=True)

    combined_df['E.coli count (CFU/g)'] = combined_df['E.coli count (CFU/g)'].str.replace(',',
                                                                                          '').astype(int)

    return combined_df


def create_agent(df):
    # Initializing the agent
    agent = create_pandas_dataframe_agent(
        model, df, verbose=True, return_intermediate_steps=True)
    return agent


st.title('LangChain Pandas DataFrame Agent Interface for SG Chicken Rice Dataset')
st.write('Vertex AI LLM model used: ', model.model_name)

# User upload
uploaded_files = st.file_uploader("Choose a CSV file", type=[
                                  "csv", "txt", "xls", "xlsx"], accept_multiple_files=True)

if uploaded_files:
    df = load_data(uploaded_files)
    # Create a string buffer
    buffer = StringIO()

    # Save dataframe info to buffer
    df.info(buf=buffer)

    # Retrieve the string value
    s = buffer.getvalue()

    # Display dataframe info
    st.text('Dataframe Info:')
    st.text(s)

    agent = create_agent(df)

    question = st.text_input('Enter your question:')
    if st.button('Ask'):
        response = agent({"input": question})
        answer = response["output"]
        st.write(answer)
        verbose_output = response["intermediate_steps"]
        st.text_area('Verbose Intermediate Output:',
                     value=verbose_output, height=200)
        intermediate_steps = response["intermediate_steps"]

        if intermediate_steps:
            # Change to -1 for the last step
            action, result = intermediate_steps[0]
            st.text(action.log)  # Display the log message
            if action.tool == 'python_repl_ast':
                try:
                    # Create a string buffer
                    buffer = StringIO()

                    # Redirect stdout to the buffer
                    sys.stdout = buffer

                    # Remove markdown syntax from tool_input
                    action.tool_input = action.tool_input.replace(
                        "```", "").strip()

                    exec(action.tool_input)

                    # Reset stdout
                    sys.stdout = sys.__stdout__

                    # Retrieve the string value from the buffer
                    s = buffer.getvalue()

                    # Display the stdout content
                    st.text(s)

                    # After executing the action, capture the current figure
                    fig = plt.gcf()
                    # If the figure is not empty, display it
                    if fig.axes:
                        st.pyplot(fig)
                    # Clear the figure so it doesn't interfere with subsequent plots
                    plt.clf()
                except Exception as e:
                    st.text(f"Error executing action: {str(e)}")
            # Display result if it's a DataFrame or a Series
            if isinstance(result, (pd.DataFrame, pd.Series)):
                st.dataframe(result)

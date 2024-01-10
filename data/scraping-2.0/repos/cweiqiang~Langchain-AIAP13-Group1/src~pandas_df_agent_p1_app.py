'''
To run this streamlit app, run the following command in your terminal:
streamlit run src/pandas_df_agent_p1_app.py

Input your question in the text box and click 'Ask' to get an answer.

Download csv file from http://www-stat.stanford.edu/~tibs/ElemStatLearn/datasets/SAheart.data

Sample Questions for input
Q1: OPENAI: "What is the shape of the dataset?
# PALM API: What is the shape of the dataset?

Q2: OPENAI: "How many missing values are there in each column?""
# PALM API: How many missing values are there in each column?

Q3: OPENAI: "Display 5 records in form of a table.
# PALM API: Not completely successful with PALM API, here are the attempts that resulted in only output of 1st row
# Prompt 1: Print the first 5 records in form of a table.
# Prompt 2: Print the first 5 rows in form of a table.
# Prompt 3: Print the first 5 rows of dataframe in form of a table.
# Prompt 4: Print the first five rows of dataframe in form of a table. Take the whole of the observations

Q4. OPENAI: "Show the distribution of people suffering with chd using bar graph."
# PALM API: Show the distribution of people suffering with chd using bar graph. Show only 0, 1 for chd on the x-axis

Q5: OPENAI: "Show the distribution of age where the person is
suffering with chd using histogram with
0 to 10, 10 to 20, 20 to 30 years and so on."
)
# PALM API: Show the distribution of age where the person is suffering with chd using histogram with 10 bins for age

Q6: OPENAI: "Draw boxplot to find out if there are any outliers
in terms of age of who are suffering from chd."
# It doesnt work well on PALM VertexAI, more prompt engineering is needed!!
# PALM API: Draw boxplot in terms of age, by chd

Q7: OPENAI: "validate the following hypothesis with t-test.
Null Hypothesis: Consumption of Tobacco does not cause chd.
Alternate Hypothesis: Consumption of Tobacco causes chd."

# PALM API: validate the following hypothesis with t-test. Null Hypothesis: Consumption of Tobacco does not cause chd. Alternate Hypothesis: Consumption of Tobacco causes chd.
# beware of infinite loop of cannot from scipy.stats import ttest_ind

Q8: OPENAI: "Plot the distribution of age for both the values
of chd using kde plot. Also provide a lenged and
label the x and y axises."

# PALM API: Plot the distribution of age for both the values of chd using kde plot with pandas,  don't use hue. Also provide a lenged and label the x and y axises.


Q9: OPENAI: "Draw a scatter plot showing relationship
between adiposity and ldl for both categories of chd."

# PALM API: Draw a scatter plot showing relationship between adiposity and ldl for both categories of chd, with a colormap for chd, with color intensity shown

Q10: OPENAI: "What is the correlation of different variables with chd"
# PALM API: drop the object columns, and find What is the correlation of different numeric (float64 or int64) variables with chd"


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

    return combined_df


def create_agent(df):
    # Initializing the agent
    agent = create_pandas_dataframe_agent(
        model, df, verbose=True, return_intermediate_steps=True)
    return agent


st.title('LangChain Pandas DataFrame Agent Interface for Heart Disease Dataset')
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

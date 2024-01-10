import streamlit as st
import openai
import os
import datetime
from datetime import datetime
import cx_Oracle
import pandas as pd
import matplotlib.pyplot as plt  # Import Matplotlib
import plotly.express as px  # Import plotly.express


os.environ['TNS_ADMIN'] = 'C:\\Wallet_XDB'

openai.api_key = os.environ.get('OPEN_AI')

# Or raise a custom error message if the environment variable is not set
if openai.api_key is None:
    raise ValueError('Environment variable OPEN_AI not set')    
    
SYSTEM_PROMPT = {
    "role": "system",
    "content": """As an AI trained in SQL and Oracle Database 19c Enterprise Edition Release 19.0.0.0.0 - Production management, you are provided with the schema of a table named 'clients_2023'. Your task is to interpret user questions about the data in this table and generate precise SQL queries that answer those questions using Oracle SQL syntax.

Here's the schema of 'clients_2023' and the field type:

- CLIENTFIRSTNAME (First Name of the Client) VARCHAR2(50): This column hold the client name
- CLIENTLASTNAME (Last Name of the Client) VARCHAR2(50): This column hold the client name
- LANGUAGE VARCHAR2(50)
- CITY (City of the Client) VARCHAR2(50)
- PROVINCE (Province of the Client) VARCHAR2(50)
- POSTALCODE (Postal Code of the Client) VARCHAR2(50)
- DATEOFBIrTH (Birth Date of the Client) DATE (Format YY-MM-DD)
- GENDER VARCHAR2(50)
- MARITALSTATUS (Marital Status of the Client) VARCHAR2(50)
- remboursement (Reimbursement) VARCHAR2(50)
- AMOUNTDUE (Amount Owed) NUMBER: The tax amount the client owe

When the user asks a question, follow these steps:

1. Identify the subject of the question (e.g., "client", "gender", "amount owed").
2. Determine the operation required by the question (e.g., counting, listing, summing).
3. Recognize any grouping or ordering needs (e.g., by gender, city, province).
4. Formulate an SQL query that uses the correct fields and operations to answer the question.
5. If the question involves string comparisons, use the UPPER function to handle different capitalizations.

Ensure that the SQL query is syntactically correct, uses the proper field names from the table schema, and would execute successfully in an Oracle SQL environment. The query should be tailored to provide an accurate answer based on the structure and contents of the 'clients_2023' table.

For instance, if the user asks, 'How many clients do I have based on gender?', you would generate the following SQL query:

SELECT GENDER, COUNT(*) AS number_of_clients
FROM CLIENTS_2023
GROUP BY GENDER
"""
}

# Define a function to run the query and fetch the results
def run_query(query):
    password = os.environ.get('PASSWORD')
    if password is None:
        raise ValueError("Environment variable PASSWORD not set")
    connection = cx_Oracle.connect(user='admin', password=password, dsn='xdb_high')
    
    # Create a cursor object from the connection
    cursor = connection.cursor()

    cursor.execute(query)
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]  # Get column names
    cursor.close()
    connection.close()
    return pd.DataFrame(rows, columns=columns)  # Convert to DataFrame for better display


def get_response_from_chatgpt(user_prompt, temperature):
    try:
        messages = [
            SYSTEM_PROMPT,
            {"role": "user", "content": user_prompt}
        ]
        response = openai.ChatCompletion.create(
            model="ft:gpt-3.5-turbo-0613:personal::8E797F6L",    # This is your fine-tuned model
            messages=messages,  # Use 'messages' directly
            temperature=temperature  # Set the temperature
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return f"Error: {e}"

def save_feedback_to_file(feedback, user_input, ai_response):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("feedback.txt", "a") as file:
        file.write(f"Timestamp: {timestamp}\n")
        file.write(f"User Input: {user_input}\n")
        file.write(f"AI Response: {ai_response}\n")
        file.write(f"Feedback: {feedback}\n")
        file.write("-----------------------------------------------------\n")

def main():
    st.title("AskData")
    
    user_input = st.text_area("Your Question:", "")
    
    # Add a slider for temperature control
    temperature = st.slider("Set Temperature", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
    
     # Check if session state variables are set, otherwise initialize them
    if 'show_feedback' not in st.session_state:
        st.session_state.show_feedback = False
    if 'feedback_submitted' not in st.session_state:
        st.session_state.feedback_submitted = False
    
    ai_response = ""
    if st.button("Send"):
        if user_input:
            st.write(f"User: {user_input}")
            ai_response = get_response_from_chatgpt(user_input, temperature)
            st.write(f"AskData: {ai_response}")
            st.session_state.show_feedback = True  # Display the feedback section

            try:
                cleaned_response = ai_response.rstrip(';')  # Remove any trailing semicolons
                df = run_query(cleaned_response)
                if not df.empty:
                    col1, col2 = st.columns(2)  # Create two columns
                    
                    with col1:  # Display DataFrame in the first column
                        st.dataframe(df)
                    
                    with col2:  # Display graph in the second column
                        # Assuming the data can be visualized as a bar graph
                        # You may need to adjust the column names and plot code to match your data
                        fig = px.bar(df, x=df.columns[0], y=df.columns[1], title="Your Graph Title", labels={df.columns[0]: "X-axis Label", df.columns[1]: "Y-axis Label"})
                        fig.update_layout(autosize=False, width=500, height=400)
                        st.plotly_chart(fig)
                        
                else:
                    st.write("No records found.")
            except Exception as e:
                st.write(f"An error occurred while executing the query: {str(e)}")
                #st.write(f"")    

            
    # Collect user feedback on the response
    if st.session_state.show_feedback and not st.session_state.feedback_submitted:
        feedback = st.radio("Was this response helpful?", ["Yes", "No"])
        if st.button("Submit Feedback"):
            save_feedback_to_file(feedback, user_input, ai_response)
            st.write(f"Thank you for your feedback!")
            st.session_state.feedback_submitted = True  # Hide the feedback section

    if st.button("Clear Chat"):
        st.session_state.show_feedback = False
        st.session_state.feedback_submitted = False
        st.empty()


if __name__ == "__main__":
    main()



import streamlit as st
import openai
import os
import csv
from io import StringIO

# Set your OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Open AI - Chat GPT API
def extract_table_details(sql_script):
    # prompt1 = f"Given a complex SQL script:\n{sql_script}\n\nPlease provide details about each table used, including the table name, column names, and any corresponding filter conditions for each column. Organize the output in a tabular form and save it into a CSV file."
    prompt = f"Given a complex SQL script:\n{sql_script}\n\n Provide details about each table used, including the table name, column names, any corresponding filter conditions for each column, and an indicator if the table is a derived table. Present the information in a tabular form with the schema 'table_name | column_name | column_filter_condition | derived_table_indicator.'  "
    # prompt=f"The following is a complex SQL script called sql_script. Provide details about each table used, including the table name, column names, any corresponding filter conditions for each column, and an indicator if the table is a derived table. Present the information in a tabular form with the schema 'table_name | column_name | column_filter_condition | derived_table_indicator.' ---------------- {sql_script}",

    messages = [{"role": "user", "content": prompt}]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.0
        )
        message_text = response['choices'][0]['message']['content']
        return message_text
    except Exception as e:
        print("Error in extract_table_details:", e)  # Add this line
        return

def main():
    st.title("SQL Details Extraction with OpenAI GPT-3")

    # Upload SQL script file
    uploaded_file = st.file_uploader("Upload a SQL script file", type=["txt"])

    if uploaded_file is not None:
        # Read file content
        sql_script_content = uploaded_file.read().decode('utf-8')

        # # Display SQL script content
        # st.subheader("Uploaded SQL Script Content:")
        # st.text(sql_script_content)

        # Button to generate details
        if st.button("Process input"):
            with st.spinner("Extracting Table Details..."):
                # Get details from OpenAI API
                details_text = extract_table_details(sql_script_content)

                # Display details
                st.subheader("Table Details:")
                st.text(details_text)

                output_filename = "table_details.txt"
                # Save details to a text file
                with open(output_filename, "w", encoding="utf-8") as file:
                    file.write(details_text)

                # Provide download link for the text file
                st.subheader("Download Table Details Text:")
                st.download_button(
                    label="Download Text",
                    data=details_text,
                    file_name=output_filename,
                    key="download_button"
                )

if __name__ == "__main__":
    main()

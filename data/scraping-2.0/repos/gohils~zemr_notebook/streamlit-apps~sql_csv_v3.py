import streamlit as st
import openai
import os
import csv
from io import StringIO

# Set your OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

def convert_to_csv(tabular_data, schema, csv_file_name):
    # Convert the tabular data string to a list of rows
    rows = [row.split('|') for row in tabular_data.strip().split('\n')]

    # Convert the schema dictionary to a list of field names
    fieldnames = list(schema.values())

    # Create a StringIO object to simulate a file in memory
    csv_buffer = StringIO()

    # Create a CSV writer with the specified field names
    csv_writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)

    # # Write the header to the CSV file
    # csv_writer.writeheader()

    # Iterate through rows, convert to dictionary using schema, and write to CSV
    for row in rows:
        row_dict = {fieldnames[i]: value.strip() for i, value in enumerate(row)}
        csv_writer.writerow(row_dict)

    # Save the CSV content to the file
    csv_data = csv_buffer.getvalue()
    with open(csv_file_name, 'w', newline='', encoding='utf-8') as csv_file:
        csv_file.write(csv_data)

    return csv_data

# Open AI - Chat GPT API
def extract_table_details(sql_script):

    # prompt = f"Given a complex SQL script:\n{sql_script}\n\n Provide details about each table used, including the table name, column names, any corresponding filter conditions for each column, and an indicator if the table is a derived table. Present the information in a tabular form with the schema 'table_name | column_name | column_filter_condition | derived_table_indicator.'  "
 
 
    prompt = f"""
 Given the following SQL script:

-- SQL script here ----- \n{sql_script}\n\n 

---
List details about all tables and columns used, including the table name, column names, any corresponding filter conditions for each column, the repeating joining column in each table, the joining condition, and the table type. Present the information in a tabular text file with pipe delimiter with the schema 'table_name  | column_name | column_filter_condition | table_type | sql_condition.'

This table should include information about base tables, derived tables, temporary tables, and common table expressions (CTEs). Additionally, include join conditions in the sql_condition.      """

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

                print("=================\n",details_text)
                # # Display details
                # st.subheader("Table Details:")
                # st.text(details_text)

                # output_filename = "table_details.txt"
                # # Save details to a text file
                # with open(output_filename, "w", encoding="utf-8") as file:
                #     file.write(details_text)


                schema = {
                    'table_name': 'table_name',
                    'table_Type': 'table_Type',
                    'column_name': 'column_name',
                    'column_filter_condition': 'column_filter_condition',
                    'Comment_Column' : 'Comment_Column'
                }

                output_csv_file_name = 'sql_output.csv'

                csv_data = convert_to_csv(details_text, schema, output_csv_file_name)

                st.text(csv_data)

                # Provide download link for the text file
                st.subheader("Download CSV SQL output")
                st.download_button(
                    label="Download csv",
                    data=csv_data,
                    file_name=output_csv_file_name,
                    key="download_button"
                )

if __name__ == "__main__":
    main()

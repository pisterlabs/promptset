import json
import pandas as pd
import openai
import os
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Load the Excel file
xls = pd.ExcelFile('table1.xlsx')

# For each sheet in the Excel file
for sheet_name in xls.sheet_names:
    # Read the sheet into a DataFrame
    df = xls.parse(sheet_name, header=None)

    # Write the DataFrame to a CSV file
    # df.to_csv(f'{sheet_name}.csv', index=False, header=False)

    # Add a column before the first column which contains the number of the row
    df.insert(0, 'row', range(1, 1 + len(df)))

    # Only keep the first and the last 10 rows of the DataFrame
    df = pd.concat([df.head(10), df.tail(10)])

    # Retrieve start row, end row and the context of the table by calling the OpenAI API
    prompt = f"""
The provided dataset was converted from XLSX to CSV. It may or may not contain any rows before or after the actual data set.

Below, you will find the first and last rows of the CSV. Determine if any rows are not part of the dataset, but are instead titles or other information that should not be part of a relational database. 
Totals or summaries should not be part of the dataset. 
Header rows that directly refer to columns should be part of the dataset.
Everything that is not part of the dataset should be kept "context" in your response.

Respond only with a JSON object that includes the keys "startRow", "endRow", "context". 
"startRow" indicates where the dataset starts (not including the context), "endRow" indicates where it ends. "context" should be a summary of the content of the excluded rows.

Table:

""" + df.to_csv(index=False, header=False)
    print(prompt)

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a data analyst who is tasked with cleaning a dataset."},
            {"role": "user", "content": prompt}
        ]
    )
    print(response.choices[0].message.content)

    response = response.choices[0].message.content
    response = response[response.find('{'):response.rfind('}')+1]
    response = json.loads(response)

    start_row, end_row, context = response['startRow'], response['endRow'], response['context']
    print(start_row, end_row, context)

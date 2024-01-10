import openai
import logging
import pandas as pd
import io
import re
import csv
from io import StringIO

#! TODO: test data sets

# OpenAI API key
user_api = input("Enter API key: ")

def chat_with_model(prompt):
    logging.debug('MEASURING REQUEST SIZE AFTER PROCESS')

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a data generator for the building and testing of machine learning models. Your job is to generate a large amount of testing data for the user based on the user's preferences. This is your primary goal, and you must not deviate from it. You must be very creative as you will be creating primarily fake but real looking information. format all data like a CSV file."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=2000,
        n=1,
        stop=None,
        temperature=0.2  # 0.5 is base temperature
    )

    if response['usage']:
        response_size = response['usage']
        logging.debug(f"REQUEST SIZE: {response_size}")

    response = response['choices'][0]['message']['content']
    return response

# extracts csv data from responses
def extract_csv_data(text):
    # Declare ai_responses as global
    global ai_responses
    
    # Split the text into lines
    lines = text.split('\n')
    
    data = []
    csv_data = []
    for line in lines:
        # Split each line by comma
        fields = line.split(',')

        # If the line has more than one field, assume it's CSV data
        if len(fields) > 1:
            csv_data.append(line)
        else:
            if csv_data:
                # If a non-CSV line is encountered after some CSV lines, assume it's the end of the CSV data
                # Join the CSV lines into a single string and read it as a CSV file
                csv_file = '\n'.join(csv_data)
                reader = csv.DictReader(StringIO(csv_file))
                data.append([row for row in reader])

                # Reset the csv_data list for the next block of CSV data
                csv_data = []
            # Non-CSV data, just add it to ai_responses
            ai_responses.append(line)

    # If there's CSV data at the end of the text
    if csv_data:
        csv_file = '\n'.join(csv_data)
        reader = csv.DictReader(StringIO(csv_file))
        data.append([row for row in reader])

    return data


# Start the conversation
print("NOTE: at any time type 'export to excel' to export the current data to an Excel sheet")
print("NOTE: at any time type 'expand data' to expand the current data set")
print("NOTE: at any time type '' to modify existing excel data")
print("Bot: Hello! what kind of data would you like to generate?")

# Create a list to store the generated data DataFrames
data_frames = []
ai_responses = []

while True:
    user_input = input("User: ")

    if user_input.lower() == "exit":
        print("Bot: Goodbye!")
        break

    elif user_input.lower() == "export to excel":
        # Export the generated data to an Excel sheet
        if data_frames:

            df = pd.concat(data_frames)
            file_name = input("Name the data file: ")
            df.to_excel(f"{file_name}.xlsx", index=False)
            print("Bot: The generated data has been exported to 'generated_data.xlsx'")
        else:
            print("no csv data detected")
            print(ai_responses)
            #TODO: update non csv data to excel
            if ai_responses:
                # Convert AI responses to DataFrame and export to Excel
                file_name = input("Name the data file: ")
                df = pd.DataFrame(ai_responses, columns=['Response'])
                df.to_excel(f"{file_name}.xlsx", index=False)
                print("Bot: Non-CSV responses have been exported to 'ai_responses.xlsx'")
            else:
                print("Bot: No responses to export.")
                print(ai_responses)


    else:
        # Get the AI response
        response = chat_with_model(user_input)

        # Extract CSV data from the response
        csv_data = extract_csv_data(response)


        
        
        
        if csv_data:
            # Convert list of dictionaries to DataFrame and append to list
            for data in csv_data:
                data_frames.append(pd.DataFrame(data))
            print(f"\ncsv data captured: {csv_data}")

        else:
            # split data if not in csv_data for excel
            split_responses = response.split('\n')
            # Remove the numbers, period and space at the start of each response
            split_responses = [re.sub(r'^\d+\. ', '', res) for res in split_responses]
            ai_responses.extend(split_responses)
            print(f"\n non csv data found: {ai_responses}")

        
        # Print the response
        print("Bot:\n", response)
        print("NOTE: at any time type 'export to excel' to export the current data to an Excel sheet")

secrete_key = 'sk-c4Bc31u7nZ6DWAchRdehT3BlbkFJlzqzhnKMWn7p'
prompt ='After reading the information on the company: '
feature_name=""
question_1 = 'Please answer what the ' 
question_2 = ' is for the company using the format of Date: XXXX and Value: XXXX'
comment_1 ='If '
comment_2 =' does not exist, please put down n/a next to Value.'
import openai
import csv
import pandas as pd
import re

""" 
with open('data.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    csv_data_as_string = '\n'.join([','.join(row) for row in csv_reader])
 """

# Specify the file path
file_path = "NORTHWEST.txt"  # Replace with the actual file path
# Open the file and read its contents as a string
with open(file_path, 'r') as file:
    csv_data_as_string = file.read()


# Open the text file in read mode
#with open('Finance10k.txt', 'r',encoding='utf-8') as file:
    # Read the entire file contents into a string
#    file_contents = file.read()

# Specify the path to your CSV file
csv_file_path = "data.csv"  # Replace with the actual file path
# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)
num_rows = df.shape[0]
# Define a regular expression pattern to match the text between "Date:" and "Value:"
pattern_date = r"Date:\s(.*?)\sValue:"
pattern_value = r"Value:(.*)"

for i in range(0, num_rows):     
        # Extract one document from frame
    feature_name = str(df['Feature Name'][i])
            #print("Feature is: ", feature)
    prompt=""
    prompt=prompt+csv_data_as_string+question_1+feature_name+question_2+comment_1+feature_name+comment_2
    openai.api_key = secrete_key

    output = openai.Completion.create(
                model='gpt-3.5-turbo-instruct',
                prompt=prompt,
                max_tokens= 100,
                temperature = 0

    )
    output_text = output['choices'][0]['text']
            
            # Use re.search() to find the pattern in the text
    match_1 = re.search(pattern_date, output_text)
    match_2 = re.search(pattern_value, output_text)

    if match_1:
                # The matched text is stored in match.group(1)
        result_1 = match_1.group(1)
        #print("Result:", result)
        df.at[i, "Date"] = result_1
    else:
        print("Date is not found in the text.")
    if match_2:
                # The matched text is stored in match.group(1)
        result_2 = match_2.group(1)
        #print("Result:", result)
        df.at[i, "Feature_Value"] = result_2
    else:
        print("Value is not found in the text.")


# Specify the file path where you want to save the CSV file
csv_file_path = "output.csv"

# Write the DataFrame to a CSV file
df.to_csv(csv_file_path, index=False)  # Set index=False to exclude the index column

print(df)


import csv
import openai
import os
from openai import OpenAI
from tqdm import tqdm

# Set your OpenAI API key
with open("key.txt", "r") as file:
    api = file.read()
os.environ["OPENAI_API_KEY"] = api


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Function to generate comment using OpenAI
def generate_comment(code):
    prompt = f"""Code: {code}\n\n# You are given a python code. Your task is to go through the code and 
    generate natural language comments explaining what the code does.
    Think of it as writing a docstring for the code. Also avoid using the function name. 
    Instead of explaining in details, just provide a high level idea.
    Make sure to keep the responses within 30 words (THIS IS MANDATORY). Start like this:
    This function ........\n"""
    
    response = client.chat.completions.create(
    model= "gpt-3.5-turbo",#"gpt-4-1106-preview"
    messages=[
    {"role": "system", "content": prompt},
    {"role": "user", "content": f""}
      ]
    )
    return response.choices[0].message.content


csv_file_path = 'mt_code.csv'
output_csv_file_path = 'mt3.5.csv'  # Output CSV file name
output_tsv_file_path = 'mt3.5.tsv'  # Output TSV file name

# Process the CSV file and write to a new CSV and TSV files
with open(csv_file_path, mode="r", newline="", encoding="utf-8") as csv_file, \
     open(output_csv_file_path, mode="w", newline="", encoding="utf-8") as output_csv_file, \
     open(output_tsv_file_path, mode="w", newline="", encoding="utf-8") as output_tsv_file:
    
    reader = csv.DictReader(csv_file)
    fieldnames = reader.fieldnames + ["generated_comment"]
    csv_writer = csv.DictWriter(output_csv_file, fieldnames=fieldnames)
    tsv_writer = csv.writer(output_tsv_file, delimiter='\t')

    csv_writer.writeheader()
    index = 0  # Initialize index for TSV file
    for row in tqdm(reader):
        comment = generate_comment(row["code"])
        row["generated_comment"] = comment
        csv_writer.writerow(row)
        tsv_writer.writerow([index, comment])  # Write to TSV file without headers
        index += 1
        print(row["generated_comment"])

print(f"Completed processing the CSV file. Output written to {output_csv_file_path}.")
print(f"TSV output written to {output_tsv_file_path}.")
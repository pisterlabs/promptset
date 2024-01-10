# return only text which discussed battery manufacturing projets. 

import csv
from io import StringIO
from openai import OpenAI
import os

company = 'norsun'

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Read the contents of your text file
file_contents = read_file('/Users/ben/Documents/bruegel/DATAn/WORKING/TUONE/tuone/article_scrape/output/company/{}.txt'.format(company))

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="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",
)

completion = client.chat.completions.create(
  model="gpt-3.5-turbo-1106",
  messages=[
    # {"role": "system", "content": f"Content:  {file_contents} "},
    # {"role": "user", "content": f"""Format your response as a csv file. List all {company} battery manufacturing projects located in Europe. Return the following project-specific information
    #  project name, 
    #  company,
    #  country, 
    #  specific location, 
    #  status, which can be: announcement, investment decision, under construction, or operational, 
    #  cell technology,
    #  value chain stage, which ranges from raw material refining to module assembly, 
    #  capital investment (in monetary units), 
    #  manufacturing capacity (in energy units). 
    #  """}
    {"role": "system", "content": f'''
     You are an energy investment analyst. Your task is to read news article concerning solar manufacturing projects run by {company} and identifty projects the company is running. Extract the following information on all projects you find: 
     1. project name,
     2. company operating the project,
     3. country where plant is located, 
     4. city or area where plant is located, 
     5. status of the plant, which can be: announcement, investment decision, under construction, or operational, 
     Return structured comma separated value data, for the items above.  
     '''},
    {"role": "user", "content": f"""{file_contents}"""}
  ]
)

# 6. cell technology,
# 7. value chain stage, which ranges from raw material refining to module assembly, 
# 8. capital investment (in monetary units), 
# 9. manufacturing capacity (in energy units).

#print(completion.choices[0].message)

data = completion.choices[0].message.content
lines = data.split('\n')
header = [column.strip() for column in lines[0].split(',')]
data = [line.split(',') for line in lines[2:]]

# Create a CSV output stream using StringIO
output = StringIO()
csv_writer = csv.writer(output)

# Write the header and data to the CSV file
csv_writer.writerow(header)
csv_writer.writerows(data)

# Save the CSV data to a file
folder_path = 'gpt_analyse/output'
csv_filename = os.path.join(folder_path, "{}.csv".format(company))
with open(csv_filename, 'w', newline='') as csv_file:
    csv_file.write(output.getvalue())

print(f"Data written to {csv_filename}")
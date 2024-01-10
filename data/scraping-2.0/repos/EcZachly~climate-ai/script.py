import openai
import csv
import os
from schema.schemas import actor_schema, actions_schema, emissions_agg_schema, datasource_schema, target_schema

openai.api_key = os.environ.get("OPENAI_KEY")
CURRENT_YEAR = 2023
file_name = 'unfccc-data/New Zealand/Table1s1-Table 1.csv'


sample_rows = []
with open(file_name, 'r') as file:
    reader = csv.reader(file)
    all_lines = []
    for row in reader:
        all_lines.append(row)
    sample_rows = all_lines[0:9]
    sample_rows.append(all_lines[-2])
    sample_rows.append(all_lines[-1])


sample_rows_string = ""
for row in sample_rows:
    sample_rows_string += '\n'.join(row)

start_prompt = f"""
Using this sample input data, generate Python code in markdown 
that reads in the file {file_name} and maps to this schema
Use the name without extension {file_name} + year for the datasource_id
Use the file name for the name field
""" + sample_rows_string

additional_prompts = """
Find the actor name in the folder name of the file path and it's a country
Query the api https://openclimate.openearth.dev/api/v1/search/actor with query parameter name to get actor_id. 
The response from the api looks like {
data: [ {actor_id:''} ]
}
"""

# Used to process data from https://unfccc.int/reports
country_emissions_prompt = f"""
    Here are the instructions for finding the right data:
        The first few rows may be invalid
        Total_emissions should be in a total column
        Use math.floor function to convert emissions to integers
        Year field is hardcoded to {CURRENT_YEAR}
        The emissions_id is the file name plus the year
"""

output_prompt = " output to a csv file with header named {}.csv"

prompts = {
    'DataSource': ' '.join([start_prompt, datasource_schema, output_prompt.format("datasource")]),
    'Actor': ' '.join([start_prompt, additional_prompts, actor_schema, output_prompt.format("actor")]),
    'EmissionsAgg': ' '.join([start_prompt, emissions_agg_schema, additional_prompts, country_emissions_prompt,
                              output_prompt.format("emissions_agg")]),
    'Target': ' '.join([start_prompt, additional_prompts, target_schema, output_prompt.format("target")]),
    'Action': ' '.join([start_prompt, additional_prompts, actions_schema, output_prompt.format("action")]),
}

for (schema_name, prompt) in prompts.items():
    print('running for prompt: ' + schema_name)
    print(prompt)
    values = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": """
            You are a data analyst tasked with mapping a bunch of dirty CSV files to a conformed schema. 
            """},
            {"role": "user", "content": prompt}
        ],
        max_tokens=4096
    )
    output_message = values["choices"][0]["message"]["content"].split('```')
    valid_messages = filter(lambda x: x.startswith('python'), output_message)
    valid_messages = map(lambda x: x[6:], valid_messages)

    # Open the file with write permissions
    with open('output/' + file_name.split('/').pop() + '.' + schema_name + '.py', 'w') as file:
        # Write some data to the file
        file.write('\n'.join(valid_messages))

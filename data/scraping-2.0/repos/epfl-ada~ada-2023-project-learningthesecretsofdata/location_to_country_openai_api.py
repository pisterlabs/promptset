"""Script allowing to create a mapping between locations and countries with GPT"""

import ast

import pandas as pd
from openai import OpenAI

# Set the list of locations to map
location_list = ['<list of your locations to map>']

# Set your personal OpenAI API key
client = OpenAI(
    api_key='<API-KEY>'
)

# Make your request to GPT
completion = client.chat.completions.create(
    model="gpt-4",  # choose your model and check the corresponding pricing please
    messages=[
        {"role": "system",
         "content": "Only return the asked information without phrasing or intorducing your answer. Return answer in "
                    "the python variable asked. Be sure to return a variable as long as the given entry."},

        {"role": "user",
         "content": f"Map using a python dictionary each location (meaning each variable of the following list) to "
                    f"its corresponding country. Please be consistent over the country name used, i.e. if multiple "
                    f"location refer to USA please mapped them all to USA. {location_list}"}
    ]
)

# Create the mapping dictionary
location_dict = ast.literal_eval(f"{'{' + completion.choices[0].message.content + '}'}")

# Convert the dict to a pandas Dataframe for easy manipulation
location_to_country = pd.DataFrame(list(location_dict.items()), columns=['location', 'country'])

# Store information in a .csv file in the computer disk
location_to_country.to_csv('mapping_locations_to_country.csv', index=False)

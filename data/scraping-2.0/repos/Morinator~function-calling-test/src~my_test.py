import json

import openai
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")

st.title("Talk to Xentral")

query = st.sidebar.text_area("Your query")

my_custom_functions = [
    {
        'name': 'set_ui_theme',
        'description': 'Select between "Light-mode" and "Dark-mode" for the UI theme',
        'parameters': {
            'type': 'object',
            'properties': {
                'color_theme': {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["light", "dark"]
                    },
                    'description': 'The color theme for the UI'
                },
            }
        }
    },
    {
        'name': 'set_font_size',
        'description': 'Set the font-size for the UI. Possible choices are "small", "medium", and "large"',
        'parameters': {
            'type': 'object',
            'properties': {
                'font_size': {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["small", "medium", "large"]
                    },
                    'description': 'The size of the font for the UI, out of 3 possible choices'
                },
            }
        }
    },
    {
        'name': 'set_currency',
        'description': 'Set the used currency for the application. Possible choices are "USD", "EUR", and "CHF"',
        'parameters': {
            'type': 'object',
            'properties': {
                'currency_symbol': {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["USD", "EUR", "CHF"]
                    },
                    'description': 'Currency symbol for the desired currency'
                },
            }
        }
    },
    {
        'name': 'create_items',
        'description': 'Inserts the specified number of items into the product catalog, specified by product_id',
        'parameters': {
            'type': 'object',
            'properties': {
                'product_id': {
                    "type": "number",
                    'description': 'The ID of the product to be inserted'
                },
                'num_items': {
                    "type": "number",
                    'description': 'How many instances of the product to be inserted'
                }
            }
        }
    },
    {
        'name': 'get_product_info',
        'description': 'Print the product information for the specified product_id',
        'parameters': {
            'type': 'object',
            'properties': {
                'product_id': {
                    "type": "number",
                    'description': 'The ID of the product we want to get information about'
                },
            }
        }
    },
    {
        'name': 'go_to_analytics_module',
        'description': 'Show the analytics module in the application',
        'parameters': {
            'type': 'object',
            'properties': {

            }
        }
    },
    {
        'name': 'open_support_chat',
        'description': 'Open the support chat in the application to talk to a customer support agent',
        'parameters': {
            'type': 'object',
            'properties': {

            }
        }
    }
]

parameters = [x['parameters']['properties'] for x in my_custom_functions]
functions_df = pd.DataFrame(
    {
        'name': [x['name'] for x in my_custom_functions],
        'description': [x['description'] for x in my_custom_functions],
        'parameters (0, 1 or multiple)': parameters
    }
)

st.table(functions_df)

openai_response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    messages=[{'role': 'user', 'content': query}],
    functions=my_custom_functions,
    function_call='auto'
)

function_call = openai_response['choices'][0]['message']['function_call']
function_name = function_call['name']
function_arguments = json.loads(function_call['arguments'])

st.sidebar.write('Note that the following results match exactly to the API-specification listed in the table.')
st.sidebar.write('This enables it to deterministically suggest the correct part of the Xentral-App.')

st.sidebar.write('\n# Function Call:')
st.sidebar.write(function_name)

st.sidebar.write('# Function Arguments:')
for key, value in function_arguments.items():
    st.sidebar.write("Key: " + str(key))
    st.sidebar.write("Value: " + str(value))
    st.sidebar.write("\n")

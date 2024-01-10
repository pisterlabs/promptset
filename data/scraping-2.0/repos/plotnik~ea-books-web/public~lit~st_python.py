# Write Python code
# =================
#
# .. csv-table:: Useful Links
#    :header: "Name", "URL"
#    :widths: 10 30
#
#    "OpenAI API Examples", https://platform.openai.com/examples
#    "Streamlit Input Widgets", https://docs.streamlit.io/library/api-reference/widgets
#    "reStructuredText Primer", https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
#    "PyLit Tutorial", https://slott56.github.io/PyLit-3/_build/html/tutorial/index.html
#
# Use OpenAI API to write Python code. You provide function description 
# in a natural language, OpenAI generates Python code.
#
# ::

import streamlit as st
from openai import OpenAI
import yaml

# Load LLM prompts.
#
# ::

prompts_file = "openai_helper.yml"
with open(prompts_file, 'r') as file:
    prompts = yaml.safe_load(file)

def get_prompt(name):
    for entry in prompts:
        if entry['name'] == name:
            return entry.get('note')
    return None
  

# We expect ``openai_helper.yml`` to contain an item:
#
# .. parsed-literal::
#
#   - name: python
#     note: Write Python code to satisfy the description you are provided.
#
#
# Call OpenAI API.
#
# ::

client = OpenAI()

def call_openai(text, op_code):
    prompt = get_prompt(op_code)
    if prompt == None:
        raise ValueError(f"Invalid prompt name: {op_code}")
  
    response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
            temperature=0.7,
        )
  
    choice = response.choices[0]

    st.write('---')
    st.write(choice.message.content)
    st.write(f'finish_reason: `{choice.finish_reason}`')
    st.write(response.usage)
    st.write(f'Choices: {len(response.choices)}')

# Input the description of Python code
#
# ::
    
label = "Description of Python code"
text = st.text_area(f"{label}:")

text = text.strip()

if len(text) > 0:
    call_openai(text, 'python')
else:
    st.write(f"Please provide the description above")
#This is script to generate jupyter notebook from chatGPT json file
#Usage: python chat2ipynb.py "display a map of USA with population data"

import sys
import nbformat as nbf
from nbformat.v4 import new_markdown_cell, new_code_cell
import re
from openai_service import get_chat_response
import uuid


TEST_PROMPT = "Display a map of USA with population data."

########################################
# Helper functions
########################################

#read text file into a string (for testing purposes)
def read_file(file_name):
    with open(file_name) as file:
        data = file.read()
    return data


########################################
# Main
########################################

#get input text string from command line
try:
    prompt = sys.argv[1]
except IndexError:
    prompt = TEST_PROMPT

# uncomment this line to read from file
#result = read_file("source.txt")

#Call OpenAI service to get result and store in result variable
result = get_chat_response(prompt)

#Split the string for code blocks
result_array = re.split(r"```", result)

# create a new notebook
nb = nbf.v4.new_notebook()
# add a header cell
header = "### Prompt: \n\n" + prompt + "\n\n### Response:"
nb.cells.append(nbf.v4.new_markdown_cell(header))

for index, item in enumerate(result_array):
    # if odd number of items, then it is a code block
    if index % 2 == 1:
        item = re.sub(r'^.*?\n|\n$', '', item)
        nb.cells.append(nbf.v4.new_code_cell(item))
    else:
        nb.cells.append(nbf.v4.new_markdown_cell(item))

# generate a file name
file_name = str(uuid.uuid4())
file_name = "notebook/"+file_name[:18] + ".ipynb"
print("Notebook create: ", file_name)

# Save the notebook to a file
nbf.write(nb, file_name)

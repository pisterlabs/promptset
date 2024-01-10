import streamlit as st
import nbformat
from io import BytesIO
import tiktoken
import dotenv
import os
import openai
from nbconvert import HTMLExporter
import chromadb
import pandas as pd
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from PIL import Image
import io
import base64

from prompts import PLANNER_AGENT_MINDSET

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

SUGGESTOR_SYSTEM_PROMPT = """ You are an expert bioinformatitian specialised in scRNA-seq analysis. Your answers are short and specific because you are a scientist.
You will recieve as an input 1) jupyter notebook of incomplete scRNA-seq analysis 2) and a library/tool description suggested by vector databse. You should suggest 
the next step in the analysis and write code for it, as if you were continuting this jupyter notebook. If the description of tool is absent, make suggestion without it."""
EMBEDDING_MODEL = "text-embedding-ada-002" # change this to biotech specialised model later
COLLECTION_NAME = 'scRNA_Tools'

# Query DB
def query_collection(collection, str_query, max_results, dataframe):
    results = collection.query(query_texts=str_query, n_results=max_results, include=['distances'])
    df = pd.DataFrame({
                'id':results['ids'][0],
                'score':results['distances'][0],
                'content': dataframe[dataframe.Name.isin(results['ids'][0])]['extented_desc_readme_trim'],
                'platform': dataframe[dataframe.Name.isin(results['ids'][0])]['Platform'],
                })

    return df


# OpenAI querying logic here
def get_code_suggestions(notebook_content, tool, selected_model):

    messages = [
        {"role": "system", "content": SUGGESTOR_SYSTEM_PROMPT}
    ]
    
    SUGGESTOR_CONTEXT = f"scRNA-seq analysis notebook right now: {str(notebook_content)}. \
        Description of a suggested tool to use: {str(tool)}. Give output in a format: 1) <why this tool is usefull>. \
            2) <Write code for the next step in the analysis using that tool given my current notebook>."  # Corrected variable name

    try:
        messages.append({"role": "user", "content": SUGGESTOR_CONTEXT})
        
        completion = openai.ChatCompletion.create(
            model=selected_model,
            messages=messages
        )
        
        assistant_message = completion.choices[0].message['content']  # Corrected attribute access
        messages.append({"role": "assistant", "content": assistant_message})
        return assistant_message

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

def num_tokens_from_string(string: str, model_name: str = "gpt-3.5-turbo") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
    
from nbformat.notebooknode import NotebookNode

def remove_code_cells(notebook: NotebookNode) -> NotebookNode:
    """
    Removes all code cells from a Jupyter notebook.

    :param notebook: The Jupyter notebook object.
    :return: A new notebook object with all code cells removed.
    """
    # Create a deep copy of the notebook to avoid modifying the original
    new_notebook = nbformat.v4.new_notebook()
    new_notebook.cells = [cell for cell in notebook.cells if cell.cell_type != 'code']
    return new_notebook

# Function to display an image from base64 string
def display_image(b64_string):
    image_data = base64.b64decode(b64_string)
    image = Image.open(io.BytesIO(image_data))
    st.image(image)

# Function to check if cell contains an anchor tag
def contains_anchor_tag(cell_content):
    return '<a id=' in cell_content

def display_and_accumulate_cells(nb, n=3):
    notebook_str = ""  # String to store notebook content
    total_cells = len(nb['cells'])

    # Function to append cell outputs to the string
    def append_cell_outputs(cell_outputs):
        nonlocal notebook_str
        for output in cell_outputs:
            if output['output_type'] in ['execute_result', 'display_data']:
                if 'text/plain' in output['data']:
                    output_text = output['data']['text/plain']
                    notebook_str += output_text + "\n"
            elif output['output_type'] == 'stream':
                if 'text' in output:
                    notebook_str += output['text']

    # Iterate over all cells
    for i, cell in enumerate(nb['cells']):
        # Skip cells with anchor tags
        cell_content = ''.join(cell['source'])
        if contains_anchor_tag(cell_content):
            continue

        # Accumulate cell content
        notebook_str += cell_content + "\n"

        # Append outputs of all cells to the string
        append_cell_outputs(cell.get('outputs', []))

        # Display logic for last 'n' cells
        if i >= total_cells - n:
            if cell['cell_type'] == 'markdown':
                st.markdown(cell_content)
            elif cell['cell_type'] == 'code':
                st.code(cell_content)

                # Display outputs for last 'n' cells
                for output in cell.get('outputs', []):
                    if output['output_type'] in ['execute_result', 'display_data']:
                        if 'text/plain' in output['data']:
                            st.text(output['data']['text/plain'])
                        if 'image/png' in output['data']:
                            display_image(output['data']['image/png'])
                    elif output['output_type'] == 'stream':
                        if 'text' in output:
                            st.text(output['text'])

    return notebook_str

# Load table with tools and their descriptions
tool_table_with_readmes = pd.read_csv('dataframes/tool-table-with-readmes.csv')

# Load assosiated vector DB
chroma_client = chromadb.PersistentClient()
try:
    embedding_function = OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"), model_name=EMBEDDING_MODEL)
    scrnatools_description_collection = chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_function)
except Exception as e:
    st.error(f"Error getting chroma collection: {e}")

# test Vector DB
try:
    print(query_collection(scrnatools_description_collection, 'quality controll python', 3, tool_table_with_readmes))
except Exception as e:
    print(e)

# Setup Streamlit App

st.set_page_config(layout="wide")
st.title('scRNA-seq copilot')

# Sidebar for model selection
model_options = ["gpt-3.5-turbo", "gpt-4-1106-preview", "gpt-3", "gpt-3.5"]
selected_model = st.sidebar.selectbox("Choose the OpenAI model", model_options)

with st.sidebar:
    st.write(f"You selected: {selected_model}")
    uploaded_file = st.file_uploader("Upload Jupyter Notebook", type="ipynb")

# Center: upload and display the jupyter notebook
if uploaded_file is not None:
    notebook_content = nbformat.read(uploaded_file, as_version=4)
    last_n_cells = st.slider("Show last N cells", 3, len(notebook_content['cells']), 10)
    st.markdown(f"> {len(notebook_content['cells']) - last_n_cells} .ipynb cells are hidden")
    st.divider()
    notebook_content_str = display_and_accumulate_cells(notebook_content, last_n_cells)
    notebook_content_nocode = str(remove_code_cells(notebook_content))
    st.sidebar.write(f"Notebook token size without code cells: {num_tokens_from_string(notebook_content_nocode)} tokens")
    st.sidebar.write(f"Notebook token size withoud images: {num_tokens_from_string(notebook_content_str)} tokens")

# Generate and display suggestions using tools' vector DB as a context
if uploaded_file and st.button('Generate next step in the analysis'):
    
    query_results_df = query_collection(scrnatools_description_collection, notebook_content_str, 2, tool_table_with_readmes)
    #if query_results_df:
    #    st.success('We figured out next step!', icon="✅")
    st.write(query_results_df)
    top_tools_desc, top_tools_names = query_results_df['content'].tolist() , query_results_df['id'].tolist()
    # stupidest way to query ever, gotta fix and rethink all of it
    # Create three columns for suggestions
    col1, col2 = st.columns(2)
    tools_columns = [col1, col2]

    for index, tool_desc, tool_name in zip([0, 1], top_tools_desc, top_tools_names):
        with tools_columns[index]:
            st.subheader(f'Suggestion: {tool_name}')
            suggestions = get_code_suggestions(notebook_content_str, tool_desc, selected_model)
            st.write(suggestions)

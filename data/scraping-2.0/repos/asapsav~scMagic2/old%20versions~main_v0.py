import streamlit as st
import nbformat
from io import BytesIO
import tiktoken
import dotenv
import os
import openai

from prompts import PLANNER_AGENT_MINDSET

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

SUGGESTOR_SYSTEM_PROMPT = """ You are an expert bioinformatitian. 
You take an input of a jupyter notebook and a tool description and you suggest the next step in the analysis and write code for it."""

# Placeholder function for querying the vector database
def query_vector_db(notebook_content):
    # Implement querying logic here
    # Return top 3 tools
    return  ['Tool1']#['Tool1', 'Tool2', 'Tool3']

# OpenAI querying logic here
def get_code_suggestions(notebook_content, tool=''):

    messages = [
        {"role": "system", "content": SUGGESTOR_SYSTEM_PROMPT}
    ]
    
    SUGGESTOR_CONTEXT = str(notebook_content) + str(tool)  # Corrected variable name

    try:
        messages.append({"role": "user", "content": SUGGESTOR_CONTEXT})
        
        completion = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",
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


st.title('scRNA-seq copilot')

uploaded_file = st.file_uploader("Upload Jupyter Notebook", type="ipynb")
if uploaded_file is not None:
    # Read the notebook file
    notebook_content = nbformat.read(uploaded_file, as_version=4)
    notebook_content = remove_code_cells(notebook_content)

    # Display the notebook file (You can use libraries like nbconvert to render the notebook)
    # st.write(notebook_content)
    st.write(f"Notebook size without code: {num_tokens_from_string(str(notebook_content))} tokens")

    if st.button('Generate enxt step in the analysis'):
        # Query the vector database
        top_tools = query_vector_db(notebook_content)

        # Get suggestions from OpenAI for each tool
        # Diplay them in 3 separate columns
        for tool in top_tools:
            st.subheader(f'Suggestion: {tool}')
            suggestions = get_code_suggestions(notebook_content, tool)
            st.write(suggestions)


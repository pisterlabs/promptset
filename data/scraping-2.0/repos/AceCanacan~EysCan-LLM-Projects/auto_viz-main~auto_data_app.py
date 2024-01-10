import streamlit as st
import pandas as pd
import openai
import nbformat as nbf
import base64

# Make sure you set this with your actual key
openai.api_key = 'YOUR API KEY'

def load_data(file):
    data = pd.read_csv(file)
    return data

def generate_response(prompt):
    st.session_state['messages'].append({"role": "system", "content": prompt})

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=st.session_state['messages'],
        max_tokens=13000,
        temperature=0.1
    )
    return completion.choices[0].message['content']

def get_column_info(column_name):
    st.session_state['messages'] = [{"role": "system", "content": f"In just a maximum of 30 words, describe the information that is contained in a dataset column named '{column_name}'."}]
    response = generate_response(st.session_state['messages'][-1]["content"])
    response = " ".join(response.split()[:30])
    return response

def get_dataset_description(data_head):
    data_head_str = str(data_head)

    st.session_state['messages'] = [{"role": "system", "content": f"I'm an AI trained to give a brief description of a dataset. Based on this sample of the data, what can you tell me about the dataset? {data_head_str}"}]
    response = generate_response(st.session_state['messages'][-1]["content"])
    response = " ".join(response.split()[:30])
    return response

def create_notebook(response):
    nb = nbf.v4.new_notebook()

    # Define the markdown and code cells
    cells = []

    blocks = response.split('```')  # Split blocks by ```
    
    for i, block in enumerate(blocks):
        block = block.strip()

        if block.startswith('python'):  # Check if block is a python code
            code = block[6:]  # Remove 'python' (6 characters) from the beginning
            cells.append(nbf.v4.new_code_cell(code))

        elif block:  # If block is not a python code and not an empty string
            # Check for and remove double asterisks
            if block.startswith('**') and block.endswith('**'):
                block = block[2:-2]  # Remove the first and last two characters (the asterisks)

            cells.append(nbf.v4.new_markdown_cell(block))

    nb['cells'] = cells

    # Write notebook object to file
    with open('notebook.ipynb', 'w') as f:
        nbf.write(nb, f)

    return 'notebook.ipynb'  # Return the filename

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{file_label}.ipynb">{file_label}</a>'
    return href

def main():

    st.image("vizwiz_log.png")
    st.write('Welcome to VizWiz! Your LLM-powered data visualization assistant. To begin, upload your CSV file')
    st.markdown("Developed by [Eys](https://www.linkedin.com/in/acecanacan/)", unsafe_allow_html=True)
    file = st.file_uploader("", type=['csv'])

    if file is not None:
        with st.spinner('Loading data...'):
            data = load_data(file)
            st.write(data.head())

        
        if 'messages' not in st.session_state:
            st.session_state['messages'] = [{"role": "system", "content": "I'm an AI trained to describe dataset columns and suggest data analyses along with their respective Python code."}]
        
        st.write("**These are the descriptions of your dataset and columns. They are preloaded with the AI's own interpretation, but you can edit them to your own liking.**")

        if 'file_uploaded' not in st.session_state or st.session_state['file_uploaded'] != file:
            st.session_state['file_uploaded'] = file
            st.session_state['column_info'] = {col: get_column_info(col) for col in data.columns}
            st.session_state['dataset_desc'] = get_dataset_description(data.head())

        if 'column_info' not in st.session_state:
            st.session_state['column_info'] = {}
            for col in data.columns:
                st.session_state['column_info'][col] = get_column_info(col)

        if 'dataset_desc' not in st.session_state:
            st.session_state['dataset_desc'] = get_dataset_description(data.head())
        dataset_desc_input = st.text_input('Dataset', st.session_state['dataset_desc'])

        for col in data.columns:
            st.session_state['column_info'][col] = st.text_input(f"{col}", st.session_state['column_info'][col])

        st.write("**When you're done checking the descriptions, press submit. You can press the submit button multiple times to create new outputs.**")

        if st.button('Submit'):
            with st.spinner('Generating response...'):
                suggestion_prompt = f"Dataset description: {dataset_desc_input}. Columns: {', '.join([f'{k} - {v}' for k, v in st.session_state['column_info'].items()])}. " \
                                    f"'Distribution', 'Relationship', 'Composition', 'Comparison'"\
                                    f"Hey ChatGPT your goal here is that given this set of information abut the dataset. You are tasked to create questions about it that can be answered through data visualization practices in python. Each question should have a sample code in python on how it can be inputted to a jupyter notebook."\
                                    f"It is imperative that EACH QUESTION has A SAMPLE CODE"\
                                    f"These are the types of analyses that should be outputted. At least one question will be generated per analysis type. More may be generated if possible. "\
                                    f"Make a set of questions per each type. For example, output questions that are based on distribution. And output another set of questions for relationship. "\
                                    f"I emphasize that each question needs to have its own block of code."\
                                    f"Assume that the dataset is loaded into a pandas DataFrame named 'df'. "\
                                    f"Make a block of code that will include all the codes for prerequisites. Such as importing necessary libraries and loading the file in a csv file."

                st.session_state['response'] = generate_response(suggestion_prompt)
                st.session_state['messages'] = []
                st.session_state['notebook_file'] = create_notebook(st.session_state['response'])

        if 'response' in st.session_state:   # check if response exists in the session state
            st.write(f"{st.session_state['response']}")   # print the response from the session state

        st.write("**Convert the output into a Jupyter notebook**")

        if 'notebook_file' in st.session_state:
            if st.button('Generate Download Link'):
                st.markdown(get_binary_file_downloader_html(st.session_state['notebook_file'], 'VizWiz Jupyter Notebook'), unsafe_allow_html=True)

        st.markdown('<h1 style="text-align: center;">Data Privacy</h1>', unsafe_allow_html=True)
        st.write('The data inputted by users in this application is not collected by the developer. Regarding data privacy rules for the chatbot, they adhere to the terms and regulations of OpenAI.')

if __name__ == "__main__":
    main()

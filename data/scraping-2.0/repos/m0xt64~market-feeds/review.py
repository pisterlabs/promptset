import os
import datetime
from openai import OpenAI
from dune_client.client import DuneClient
import streamlit as st
import json
import openai
import pandas as pd
import base64

# Function to read JSON data from the uploaded file
def load_template_json():
    """Function to load a template JSON file."""
    with open("template.json", "r") as file:
        template_data = json.load(file)
    return template_data

def load_json_data(uploaded_file):
    if uploaded_file is not None:
        string_data = uploaded_file.getvalue().decode("utf-8")
        return json.loads(string_data)
    return None

def save_json(data, default_filename='updated_data'):
    """Save data to a JSON file and return a link for downloading."""
    # Use feed_info['name'] as the filename, with a fallback to default_filename
    filename = f"{data['feed_info'].get('name', default_filename)}.json"

    # Replace characters not allowed in filenames
    filename = filename.replace('/', '_').replace('\\', '_')

    # Add 'reviewed' timestamp to user_info
    if 'user_info' not in data:
        data['user_info'] = {}
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    data['user_info']['reviewed'] = current_datetime

    with open(filename, 'w') as f:
        json.dump(data, f)
    with open(filename, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    href = f'<a href="data:file/json;base64,{b64}" download="{filename}">Download updated JSON file</a>'
    return href

# Function to prepare the prompt from JSON data
def prepare_prompt(data):
    # Modify this function based on your JSON structure and requirements
    prompt = f"{data['feed_info']['role']} {data['feed_info']['goal']} {data['feed_info']['audience']} {data['feed_info']['constraints']} {data['feed_info']['size']} {data['feed_info']['instructions']} {data['feed_info']['closing']}"
    return prompt

def get_data(query_id):
    dune = DuneClient.from_env()
    results = dune.get_latest_result(query_id)
    data_ = results.result.rows
    df = pd.DataFrame(data_)
    _data = df.loc[:0]
    merged_data = _data.to_json(orient='records')
    return merged_data

def main():
    st.sidebar.markdown('# Market Feeds Reviewer')
    st.sidebar.markdown("##### Review and Edit market feeds.")

    # API key input (should be done securely in production)
    api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password")
    api_dune_key = st.sidebar.text_input("Enter your Dune API key:", type="password")

    # Template button
    use_template = st.sidebar.button("Use Template")

    # Initialize data variable
    data = None

    if use_template:
        data = load_template_json()
    else:
        # File uploader
        uploaded_file = st.sidebar.file_uploader("Upload a JSON file", type=["json"])
        if uploaded_file is not None:
            data = load_json_data(uploaded_file)
    
    reviewer_nickname = st.sidebar.text_input("Your nickname:")
    if reviewer_nickname:
        # Update user_info with reviewer's nickname
        if 'user_info' not in data:
            data['user_info'] = {}
        data['user_info']['reviewed_by'] = reviewer_nickname

    # Input for reviewer's comments
    comment = st.sidebar.text_area("Reviewer's Comments", help="Enter any comments about the changes made")
    if comment:
        data['user_info']['comment'] = comment
    
    if api_key and api_dune_key:
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["DUNE_API_KEY"] = api_dune_key

        if data and 'feed_info' in data:
            feed_info = data['feed_info']
            st.markdown("### Edit Feed Info:")
            st.write("You only need to make changes between the square brackets in these fields:")
            st.write("Name, Goal, Size, Instructions")

            # Define a dictionary for widget types
            widget_types = {
                'name': 'text',
                'role': 'text',
                'goal': 'text',
                'audience': 'text_area',
                'constraints': 'text_area',
                'size': 'text',
                'instructions': 'text_area',
                'closing': 'text'
            }

            # Create editable fields with the specified widget type
            for key in ['name', 'role', 'goal', 'audience', 'constraints', 'size', 'instructions', 'closing']:
                widget_type = widget_types.get(key, 'text')  # Default to 'text' if not specified

                if widget_type == 'text':
                    feed_info[key] = st.text_input(key.capitalize(), value=feed_info.get(key, ''))
                elif widget_type == 'text_area':
                    feed_info[key] = st.text_area(key.capitalize(), value=feed_info.get(key, ''))


            if st.button("Save Changes"):
                # Check if both nickname and comment are provided
                if not reviewer_nickname or not comment:
                    st.warning("Please provide 1) Your nickname and 2) Reviwer's comments before saving.")
                else:
                    # Update the feed_info in the main data object
                    data['feed_info'] = feed_info
                    # Add reviewer's comments to user_info
                    data['user_info']['comment'] = comment
                    st.success("Changes saved successfully.")

            # Button to generate sample feeds
            if st.button("Create Sample Feeds"):
                if data:
                    query_id = data['feed_info'].get('query_id')
                    prompt = prepare_prompt(data)
                    merged_data = get_data(query_id)
                    model = data['model_info']['model']
                    temperature = data['model_info']['temperature']

                    samples = {}
                    for i in range(1, 4):
                        sample_output = execute_model(prompt, merged_data, model, temperature)
                        samples[f'sample_{i}'] = sample_output

                    data['testing'] = samples
                    st.markdown(save_json(data), unsafe_allow_html=True)
                    st.success("Sample feeds created and saved.")
                else:
                    st.error("Please ensure all data, reviewer nickname, and comments are provided.")


            prompt = prepare_prompt(data)

            query_id = data['feed_info'].get('query_id')
            if query_id:
                merged_data = get_data(query_id)
                st.write("Data:")
                st.text_area("Data", merged_data, height=150)

                if st.button("Run Model"):
                    model = data['model_info']['model']
                    temperature = data['model_info']['temperature']
                    response = execute_model(prompt, merged_data, model, temperature)
                    st.write("Model Response:")
                    st.text_area("Response", response, height=150)
            else:
                st.error("Query ID not found in JSON.")
    else:
        st.error("API keys are required.")

# Modify execute_model to accept model and temperature as arguments
def execute_model(prompt, merged_data, model, temperature):
    system_message = prompt
    client = OpenAI()
    response = client.chat.completions.create(
        model=model, 
        messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": merged_data}
            ],
        max_tokens=4000, 
        temperature=temperature)

    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    main()

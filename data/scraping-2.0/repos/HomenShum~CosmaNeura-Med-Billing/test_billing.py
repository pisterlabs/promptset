import openai
import re
import tenacity
from feature2_ccsr_categorization import search_index, ccsr_df_feather
import time
import os
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import streamlit as st
from streamlit_chat import message
from Bio import Entrez

import pandas as pd
import numpy as np
from numpy.linalg import norm
import openai
import streamlit as st
import re 
import json
from time import time, sleep

##### Datasets ############################################################################################################
with open("CPT_analysis_result1692639740.772306.json", 'r') as json_file:
    cpt_output = json.load(json_file)

with open("ICD_analysis_result1692639740.7752995.json", 'r') as json_file:
    icd_output = json.load(json_file)

sorted_cpt_billing_code_list = sorted(cpt_output, key=lambda x: x['average_score'], reverse=True)[:20]

sorted_icd_billing_code_list = sorted(icd_output, key=lambda x: x['average_score'], reverse=True)[:20]

##### Settings ############################################################################################################
# Use the OpenAI API key from secrets
openai.api_key = st.secrets["openai_api_key"]

# Pinecone Settings
os.environ['PINECONE_API_KEY'] = st.secrets["pinecone_api_key"]
os.environ['PINECONE_ENVIRONMENT'] = st.secrets["pinecone_environment"]
pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'), 
    environment=os.getenv('PINECONE_ENVIRONMENT')
)
index_name = 'langchainpdfchat-index'
embeddings = OpenAIEmbeddings()
docsearch = Pinecone.from_existing_index(index_name, embeddings)

# read from file prompts\system\dictation_note_analysis.txt
with open("patient_physician_SOAP_note_system.txt", 'r') as user_file:
    dictation_note_analysis = user_file.read()
    print(dictation_note_analysis)

# read from file prompts\user\dictation_note_analysis_user.txt
with open("patient_physician_SOAP_note_user.txt", 'r') as user_file:
    dictation_note_analysis_user = user_file.read()
    print(dictation_note_analysis_user)

##### Functions ############################################################################################################
@tenacity.retry(
    stop=tenacity.stop_after_delay(30),
    wait=tenacity.wait_exponential(multiplier=1, min=1, max=30),
    retry=tenacity.retry_if_exception_type(openai.error.APIError),
    reraise=True,
)
def gpt_completion(prompt, model='gpt-4', temp=0, stop=["<<END>>"]):
    prompt = prompt.encode(encoding="ASCII", errors="ignore").decode()
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": dictation_note_analysis },
            {"role": "user", "content": dictation_note_analysis_user},
            {"role": "user", "content": prompt},
        ],
        temperature=temp,
        stop=stop,
    )
    text = response["choices"][0]["message"]["content"].strip()
    text = re.sub("\s+", " ", text)
    return text

def analyze_dictation_note(dictation_note):
    start_time = time()

    SOAP_note_analysis = gpt_completion(dictation_note)
    result = SOAP_note_analysis.split("<<END>>")[0].strip()

    ccsr_categories_list = search_index(SOAP_note_analysis, ccsr_df_feather['Embeddings'].tolist())

    ccsr_categories_list_list = []

    for i, category in enumerate(ccsr_categories_list, start=1):
        content = category['content']
        # append both i and category
        ccsr_categories_list_list.append(f"{i}. {content}")

    end = time()
    print(f"Runtime of the program is {end - start_time}")

    return result, ccsr_categories_list_list

def search(query):
    Entrez.email = 'hshum2018@gmail.com'
    handle = Entrez.esearch(db='pubmed',
                            sort='relevance',
                            retmax='1',
                            retmode='xml',
                            term=query)
    results = Entrez.read(handle)
    return results

def fetch_details(id_list):
    ids = ','.join(id_list)
    Entrez.email = 'hshum2018@gmail.com'
    handle = Entrez.efetch(db='pubmed',
                           retmode='xml',
                           id=ids)
    results = Entrez.read(handle)
    return results

@st.cache_data
def get_abstract(paper):
    abstract = ''
    if 'Abstract' in paper['MedlineCitation']['Article']:
        abstract = paper['MedlineCitation']['Article']['Abstract']['AbstractText']
        if isinstance(abstract, list):
            abstract = ' '.join(abstract)
    return abstract

@st.cache_data
def format_to_bullet_list(message):
    # Split the message by line breaks and '- ' sequences
    lines = re.split(r'\n|- ', message)
    
    # Strip whitespace from each line and filter out any empty lines
    lines = [line.strip() for line in lines if line.strip()]
    
    # Format each line as a bullet item
    bullet_list = [f"- {line}" for line in lines]
    
    # Join the bullet items into a single string with line breaks between each item
    formatted_message = "\n".join(bullet_list)
    
    return formatted_message

def gpt3_embedding(content, engine='text-embedding-ada-002'):
    content = content.encode(encoding='ASCII',errors='ignore').decode() 
    response = openai.Embedding.create(input=content,engine=engine) 
    vector = response['data'][0]['embedding']  
    return vector

@st.cache_data
def similarity(v1, v2):
    if v1 is None or v2 is None:
        return 0.0

    norm_v1, norm_v2 = norm(v1), norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    return np.dot(v1, v2) / (norm_v1 * norm_v2)

##### Main #####################################
start_time = time()
st.sidebar.header("Recommended PubMed Articles")

st.title("Medical Billing Code Suggestion - CosmaNeura")
st.write("- Extract and highlight medical or diagnostic information from MyCatholicDoctor's sample dialogues dataset.")
st.write("- Render the highlighted information in markdown bullet point format for clarity.")
st.write("- Utilize CosmaNeura's algorithm to provide accurate medical billing code recommendations.")
st.write("- Enable a streamlined review and confirmation of suggested billing codes within the user-friendly interface.")

st.write("Please enter dictation notes below:")

### storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

# Default CPT and ICD codes
if 'selected_cpt_options' not in st.session_state:
    st.session_state['selected_cpt_options'] = []

if 'selected_icd_options' not in st.session_state:
    st.session_state['selected_icd_options'] = []

if 'ccsr_categories_list_list' not in st.session_state:
    st.session_state['ccsr_categories_list_list'] = []

# read from file prompts\input\hello.TXT
with open("SampleWriteUp.txt", 'r') as input_file:
    dictation_note = input_file.read()

input_text = st.text_area("Dictation Notes",
                            dictation_note,
                            key = 'dictation_notes',
                            height = 300)

generated_result = []
past_result = []

# Add reset chat button
if st.button('Reset Chat'):
    generated_result = st.empty()
    past_result = st.empty()


# Extract CPT options, keeping track of codes to avoid duplicates
cpt_options_dict = {}
for item in sorted_cpt_billing_code_list:
    for result in item["results"]:
        code = result['CPT Code']
        if code not in cpt_options_dict:
            cpt_options_dict[code] = result['Description']

cpt_options = [f"{code} - {desc}" for code, desc in sorted(cpt_options_dict.items())]

# Extract ICD options, keeping track of codes to avoid duplicates
icd_options_dict = {}
for item in sorted_icd_billing_code_list:
    for result in item["results"]:
        code = result['ICD Code']
        if code not in icd_options_dict:
            icd_options_dict[code] = result['Description']

icd_options = [f"{code} - {desc}" for code, desc in sorted(icd_options_dict.items())]

cpt_options = st.multiselect(
    'Select the CPT Codes and Descriptions:',
    cpt_options,  # supposed to be a list of CPT options
    help='CPT codes are used to describe medical, surgical, and diagnostic services.',
)

icd_options = st.multiselect(
    'Select the ICD Codes and Descriptions:',
    icd_options,  # supposed to be a list of ICD options
    help='ICD codes are used to record, track, and monitor diagnoses.',
)

# You can handle the selected options here
st.write(f"You selected {len(cpt_options)} CPT code(s) and {len(icd_options)} ICD code(s).")

if st.button('Analyze Dictation Notes'): 
    generated_result = st.empty()
    past_result = st.empty()
    if openai.api_key and dictation_note:  # Only run if both API key and patient notes are provided
        past_result = dictation_note
        with st.expander(f"Dictation Note", expanded=True): 
            if past_result:
                st.subheader("Dictation Notes:")
                st.write(past_result)

        with st.spinner('Analyzing...'):
            try:
                # Run GPT model to analyze patient's note
                result, ccsr_categories_list_list = analyze_dictation_note(dictation_note)

                # Update the session state for categories
                st.session_state['ccsr_categories_list_list'] = ccsr_categories_list_list

                # Format the result and terms
                formatted_highlighted_result = format_to_bullet_list(result)

                # Format the CCSR categories as a bulleted list
                formatted_ccsr_categories = "CCSR Categories Identified:\n" + "\n".join(f"- {item}" for item in ccsr_categories_list_list)

                # show output with expander and subheader
                with st.expander(f"Analyzed Result", expanded=True):
                    st.subheader("SOAP Note Analysis Result:")
                    st.write(formatted_highlighted_result)

                with st.expander(f"CCSR Categories", expanded=True):
                    st.subheader("CCSR Categories:")
                    st.write(formatted_ccsr_categories)

                st.success('Analysis successful.')
            except Exception as e:
                st.error(f"Error in analyzing dictation notes: {e}")

# Perform PubMed search for each category
if st.session_state['ccsr_categories_list_list']:
    for category in st.session_state['ccsr_categories_list_list']:
        if st.sidebar.checkbox(f'Show Articles for {category}'):
            results = search(category)
            id_list = results['IdList']
            papers = fetch_details(id_list)
            for i, paper in enumerate(papers['PubmedArticle']):
                title = paper['MedlineCitation']['Article']['ArticleTitle']
                author_list = paper['MedlineCitation']['Article']['AuthorList']
                authors = ', '.join([author.get('LastName', '') for author in author_list])
                abstract = get_abstract(paper)
                
                # Show article in sidebar
                st.sidebar.subheader(f'Title: {title}')
                st.sidebar.write(f'Authors: {authors}')
                st.sidebar.write(f'Abstract: {abstract}')

end_time = time() - start_time
print(f"Total time of analysis: {str(end_time)} seconds")

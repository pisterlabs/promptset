import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import streamlit as st
from langchain.chains import RetrievalQA
import json
from langchain.document_loaders import PyPDFLoader
import tempfile
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import boto3
from langchain.llms import Bedrock
from sentence_transformers import SentenceTransformer
from chromadb import Embeddings
from langchain.chains.question_answering import load_qa_chain
import yaml
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

NUM_QUERY = 3
with open('prompt_template.yaml', 'r') as file:
        loaded_templates = yaml.safe_load(file)

# Advanced settings
st.set_page_config(page_title='My Complex Streamlit App', layout='wide', initial_sidebar_state='expanded')

MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name):
        self._embedding_function = SentenceTransformer(model_name)
    def embed_documents(self, texts):
        embeddings = self._embedding_function.encode(texts, convert_to_numpy=True).tolist()
        return [list(map(float, e)) for e in embeddings]
    def embed_query(self, text):
        embeddings = self._embedding_function.encode([text], convert_to_numpy=True).tolist()
        return [list(map(float, e)) for e in embeddings][0]

sentence_transformer_ef = SentenceTransformerEmbeddings(model_name=MODEL_NAME)

# Initialize session state variables if they don't exist
if 'cv_file' not in st.session_state:
    st.session_state.cv_file = None

if 'job_offer_file' not in st.session_state:
    st.session_state.job_offer_file = None

# if 'processed_cv_output' not in st.session_state:
#     st.session_state['processed_cv_output'] = None

# if 'processed_job_offer_output' not in st.session_state:
#     st.session_state['processed_job_offer_output'] = None

if 'full_report' not in st.session_state:
    st.session_state['full_report'] = None


def persist_ESCO_vectorstore():
    return Chroma(persist_directory="ESCO_mpnet_db", embedding_function=sentence_transformer_ef)
def persist_JobOffers_vectorstore():
    return Chroma(persist_directory="mpnet_embeddings_db", embedding_function=sentence_transformer_ef)
    
key_mapping = {
    'job_offer_summary': 'Job Offer Summary',
    'title': 'Title',
    'department': 'Department',
    'responsibilities': 'Responsibilities',
    'educational_background': 'Educational Background',
    'experience': 'Experience',
    'skills': 'Skills',
    'technical_proficiencies': 'Technical Proficiencies',
    'regulatory_knowledge': 'Regulatory Knowledge',
    'certifications': 'Certifications',
    'commonly_sought_skills_and_qualifications': 'Commonly Sought Skills and Qualifications'
    # Add more key mappings as necessary
}

def dict_to_html(data_dict, level=0):
    html_output = ""
    for key, value in data_dict.items():
        display_key = key_mapping.get(key.lower(), key.replace('_', ' ').title())
        
        if isinstance(value, dict):
            # Details tag for nested dictionaries with a summary
            html_output += f"""
            <div style='margin-bottom: 10px;'>
                <details>
                    <summary style='font-weight: bold; color: #2c3e50; cursor: pointer;'>
                        {display_key}
                    </summary>
                    <div style='padding: 10px;'>
                        {dict_to_html(value, level+1)}
                    </div>
                </details>
            </div>
            """
        elif isinstance(value, list) and value:
            # Only create the list if it's not empty
            list_items = ''.join(f"<li>{dict_to_html(item, level+1) if isinstance(item, dict) else item}</li>" for item in value)
            html_output += f"<div><b>{display_key}</b><ul>{list_items}</ul></div>"
        elif isinstance(value, list):
            # If the list is empty, output nothing for the skills
            continue
        else:
            # Output for simple key-value pairs
            html_output += f"<div><b>{display_key}</b>: {value}</div>"
        
        # Add a line break for spacing
        html_output += "<br>"

    return html_output

llm = ChatOpenAI()

# Function placeholders
def process_cv(cv_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(cv_file.getvalue())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path, extract_images=True)
        cv = loader.load()

        prompt_template = loaded_templates['cv-parsing-prompt'].strip()
        prompt = ChatPromptTemplate.from_template(prompt_template)
            
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"cv": cv})
        processed_cv_output = json.loads(response)
        st.session_state['processed_cv_output'] = processed_cv_output

        st.session_state['cv_file'] = cv[0].page_content

        return dict_to_html(processed_cv_output)
    except Exception as e:
        return f"An error occurred while processing the CV: {e}"

def display_cv_processing():
    with left_column:
        st.header("CV Upload")
        cv_file = st.file_uploader("Choose a CV file", type=['pdf', 'png', 'jpg'], key='cv_uploader')
        if cv_file is not None:
            st.success(f"CV uploaded successfully: {cv_file.name}")
            html_content = process_cv(cv_file)
            st.markdown("## Processed CV", unsafe_allow_html=True)
            st.markdown(html_content, unsafe_allow_html=True)

def process_job_offer(job_offer_file):
    try:
        vectorstore = persist_JobOffers_vectorstore()
        retriever = vectorstore.as_retriever()

        prompt_template = loaded_templates['job-offer-processing-prompt'].strip()
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = (
            {"context": retriever, "joboffer": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        response = chain.invoke({"joboffer":job_offer_file})
        processed_job_offer_output = json.loads(response)
        st.session_state['processed_job_offer_output'] = processed_job_offer_output
        return dict_to_html(processed_job_offer_output)
    except Exception as e:
        return f"An error occurred while processing the Job Offer: {e}"    

def display_job_offer_processing():
    with right_column:
        job_offer_file = 'We are seeking a dynamic individual to be a part of our Audit department. As our auditor, your responsibilities will include planning and performing a variety of internal audits and special projects throughout the company, which include a variety of Senior Manage- ment driven assignments. These audits will include detailed assessment of operational, financial and systems activities where a thorough understanding of control processes must be obtained from a variety of people and resources.'
        if job_offer_file is not None:
            html_content = process_job_offer(job_offer_file)
            st.markdown("## Processed Job Offer", unsafe_allow_html=True)
            st.markdown(html_content, unsafe_allow_html=True)


def retrieve_esco_documents(query, NUM_QUERY=3):
    vectordb = persist_vectorstore()
    # print(query)
    # query = "job_title = Digital Marketing Specialist"
    # print('REEULTS', vectordb.similarity_search(query, NUM_QUERY))

    # Code to retrieve ESCO documents
    return [item.page_content for item in vectordb.similarity_search(query, NUM_QUERY)]



def generate_report():
    # try:
    vectorstore = persist_ESCO_vectorstore()
    retriever = vectorstore.as_retriever()
    prompt_template = loaded_templates['full-report-template'].strip()
    prompt = ChatPromptTemplate.from_template(prompt_template)
    # Select the frst experience
    # cv = json.loads(json.dumps(st.session_state['processed_cv_output']['Experiences'][0]))
    cv = st.session_state['processed_cv_output']
    job_offer = st.session_state['processed_job_offer_output']
    chain = (
        {"context": retriever, "cv": RunnablePassthrough(), "job_offer": RunnablePassthrough()}
        | prompt 
        | llm
    )
    answer = chain.invoke({"cv": cv, "job_offer": job_offer})
    return json.loads(answer.content)
    # except Exception as e:
    #     print(e)
    #     st.error(f"An error occurred while generating the report: {e}")
    #     return None

def display_full_report():
    with center_column:
        if 'processed_cv_output' in st.session_state and 'processed_job_offer_output' in st.session_state:
            st.markdown("## Generating Full Report...", unsafe_allow_html=True)
            report_content = generate_report()
            if report_content:
                st.session_state['full_report'] = report_content
                st.markdown("## Full Report Generated by LLM", unsafe_allow_html=True)
                html_content = dict_to_html(st.session_state['full_report'])
                st.markdown(html_content, unsafe_allow_html=True)      


# Main content
st.title("UPF-InfoJobs Demo App")

# Mimic two sidebars by using columns
left_column, center_column, right_column = st.columns([2, 3, 2])

# with left_column:
#     st.header("CV Upload")
#     st.session_state.cv_file = st.file_uploader("Choose a CV file", type=['pdf', 'png', 'jpg'], key='cv_uploader')
#     if st.session_state.cv_file is not None:
#         st.success("CV uploaded successfully: " + st.session_state.cv_file.name)
#         # Display a preview if it's an image
#         process_cv()
display_cv_processing()

display_job_offer_processing()

display_full_report()

# with st.container():
#     with center_column:
#         # st.header("Retrieved ESCO Documents")
#         # if st.session_state.pre_analysis_output != "":
#         #     try:
#         #         exp1 = st.session_state['pre_analysis_output']['experiences'][0]
#         #     except Exception as e:
#         #         exp1 = json.dumps(st.session_state['pre_analysis_output'])
#         #         print("Retrieved ESCO Documents --> ", e)
#         #     esco_content = retrieve_esco_documents(exp1)
#         #     st.write(esco_content)

#         # st.header("Analysis of Job Offer")
#         # if st.session_state.job_offer_file is not None:
#         #     analysis_content = analyze_job_offer()
#         #     st.write(analysis_content)
        

#         # st.header("Full Report Generated by LLLM")
#         if st.session_state['processed_job_offer'] is not None and st.session_state['processed_cv_output'] is not None:
#             generate_full_report()

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import yaml
import tempfile
import streamlit as st

from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from chromadb import Embeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough


from vis_streamlit import (plot_job_profile_match, plot_educational_qualifications, 
                           plot_skills_and_proficiencies, plot_certifications_and_regulatory_knowledge,
                           plot_spyder)


# Mock value for job profile match score (suppose the candidate's profile matches 70% with the job profile)
job_profile_match_score = 70

# Mock boolean for education match (suppose the candidate has the required educational qualifications)
education_match = True

# Mock dictionary for skills scores (each skill has a match score out of 1)
skills_scores = {
    "Logistics": 0.8,  # 80% match
    "International Trade": 0.9,  # 90% match
    "Legal Advisory": 0.7,  # 70% match
    "Fiscal Advisory": 0.5,  # 50% match
    "IT": 0.3,  # 30% match
}

# Mock list for certifications presence (True if the candidate has the certification, False otherwise)
certifications_presence = [True, False, True, False, True]

# Mock labels for certifications (corresponding to the above presence/absence list)
certifications_labels = [
    "Certified Supply Chain Professional",
    "Certified in Logistics, Transportation and Distribution",
    "Project Management Professional",
    "Has Logistics Bachelor",
    "Occupational Safety and Health Administration Certification"
]

# -------------------------------
# Setup App
# -------------------------------

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

with open('data/prompt_template.yaml', 'r') as file:
    loaded_templates = yaml.safe_load(file)

# Advanced settings
st.set_page_config(page_title='My Complex Streamlit App', layout='wide', initial_sidebar_state='expanded')
st.set_option('deprecation.showPyplotGlobalUse', False)

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


@st.cache_resource
def load_sentence_transformer(model_name):
    return SentenceTransformerEmbeddings(model_name=model_name)#SentenceTransformer(model_name)
# class SentenceTransformerEmbeddings(Embeddings):
#     def __init__(self, model_name):
#         self._model_name = model_name
#         self._embedding_function = None

#     def _load_model(self):
#         if self._embedding_function is None:
#             self._embedding_function = load_sentence_transformer(self._model_name)
    
#     def embed_documents(self, texts):
#         self._load_model()
#         embeddings = self._embedding_function.encode(texts, convert_to_numpy=True).tolist()
#         return [list(map(float, e)) for e in embeddings]

#     def embed_query(self, text):
#         self._load_model()
#         embeddings = self._embedding_function.encode([text], convert_to_numpy=True).tolist()
#         return [list(map(float, e)) for e in embeddings][0]
sentence_transformer_ef = load_sentence_transformer(MODEL_NAME)

def initialize_session_state():
    # Initialize session state variables if they don't exist
    if 'processed_cv_output' not in st.session_state:
        st.session_state['processed_cv_output'] = None

    if 'processed_job_offer_output' not in st.session_state:
        st.session_state['processed_job_offer_output'] = None

    if 'cv_file' not in st.session_state:
        st.session_state.cv_file = None

    if 'job_offer_file' not in st.session_state:
        st.session_state.job_offer_file = None

    if 'full_report' not in st.session_state:
        st.session_state.full_report = None

    if 'uploaded_files' not in st.session_state:
        st.session_state['uploaded_files'] = []

    if 'show_charts' not in st.session_state:
        st.session_state['show_charts'] = False

# Call this function at the start of your app to ensure all session state variables are initialized
initialize_session_state()

@st.cache_resource
def persist_ESCO_vectorstore():
    return Chroma(persist_directory="ESCO_mpnet_db", embedding_function=sentence_transformer_ef)

@st.cache_resource
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

@st.cache_resource
def initialize_llm():
    return ChatOpenAI()
llm = initialize_llm()

# -------------------------------
# Utility Functions
# -------------------------------

def update_processing_status(file_name, status):
    """Updates the processing status of a file."""
    status_icon = "✅" if status else "❌"
    st.write(f"{status_icon} {file_name}")


# -------------------------------
# CV Processing Functions
# -------------------------------

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
        st.error(f"An error occurred while processing the CV: {e}")

@st.cache
def get_uploaded_files():
    """Cache the uploaded files to avoid re-uploading after each rerun."""
    return []

def display_processed_cv(html_content):
    """Display the processed CV content."""
    st.markdown("## Processed CV", unsafe_allow_html=True)
    st.markdown(html_content, unsafe_allow_html=True)

# Function to update the score in the session state
def update_score(index, score):
    st.session_state['uploaded_files'][index]['score'] = score

def display_cv_processing():
    with left_column:
        with st.container():
            st.header("CV Upload")
            
            new_files = st.file_uploader("Choose a CV file", type=['pdf', 'png', 'jpg', 'jpeg'], accept_multiple_files=True, key='cv_uploader')
            if new_files:
                for new_file in new_files:
                    if not any(new_file.name == uf['file'].name for uf in st.session_state['uploaded_files']):
                        st.session_state['uploaded_files'].append({'file': new_file, 'processed': False, 'score': 1})

            for index, file_info in enumerate(st.session_state['uploaded_files']):
                with st.expander(f"{file_info['file'].name}"):
                    if not file_info['processed']:
                        if st.button('Analyze', key=f'btn_analyze_{index}'):
                            file_info['processed'] = True
                            html_content = process_cv(file_info['file'])
                            display_processed_cv(html_content)

                    if file_info['processed']:
                        # Display star ratings with visual representation
                        stars_options = [('⭐️' * i, i) for i in range(1, 6)]
                        current_score = file_info['score']
                        # We use the index of the tuple corresponding to our score
                        score_index = next((index for index, (_, score) in enumerate(stars_options) if score == current_score), 0)
                        # Display the radio buttons with stars as options
                        score = st.radio(
                            "Score", 
                            [option[0] for option in stars_options], 
                            index=score_index, 
                            key=f'score_{index}',
                            on_change=update_score, 
                            args=(index, stars_options[score_index][1])
                        )

            # CSS to style the expanders and other elements
            st.markdown("""
                <style>
                .stExpander {
                    border: 2px solid #0e1117;
                    border-radius: 0.25rem;
                    background-color: #f0f2f6;
                }
                .stExpander > .st-eb {
                    font-weight: bold;
                }
                .stButton > button {
                    color: white;
                    background-color: #0e1117;
                    border-radius: 0.25rem;
                    border: 1px solid #0e1117;
                }
                </style>
                """, unsafe_allow_html=True)
        

# -------------------------------
# Job Offer Processing Functions
# -------------------------------

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
# Streamlit interface for uploading and processing a PDF file
# def display_job_offer_processing():
#     with right_column:
#         # Streamlit file uploader
#         uploaded_file = st.file_uploader("Upload your PDF job offer", type="pdf", key='job_offer_uploader')
#         if uploaded_file is not None:
#             # Using a temporary file to read the PDF content
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#                 tmp_file.write(uploaded_file.getvalue())
#                 tmp_file_path = tmp_file.name

#             # Load the PDF file and extract text
#             loader = PyPDFLoader(tmp_file_path)
#             job_offer = loader.load()  # Adjust this line based on how your PyPDFLoader works
            
#             # Process the extracted text
#             html_content = process_job_offer(job_offer)
            
#             # Display the processed job offer
#             st.markdown("## Processed Job Offer")
#             st.markdown(html_content, unsafe_allow_html=True)


# -------------------------------
# ESCO Processing Function
# -------------------------------
# def retrieve_esco_documents(query, NUM_QUERY=3):
#     vectordb = persist_vectorstore()
    # print(query)
    # query = "job_title = Digital Marketing Specialist"
    # print('REEULTS', vectordb.similarity_search(query, NUM_QUERY))

    # Code to retrieve ESCO documents
    # return [item.page_content for item in vectordb.similarity_search(query, NUM_QUERY)]

# -------------------------------
# Full Report Processing Functions
# -------------------------------


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
    with right_column:
        st.markdown("--------------------")
        if st.session_state['processed_cv_output'] is not None and st.session_state['processed_job_offer_output'] is not None:
            with st.spinner('#### Generating Full Report...'):
                report_content = generate_report()
            if report_content:
                st.session_state['full_report'] = report_content
                st.markdown("## Full Report Generated by LLM", unsafe_allow_html=True)
                html_content = dict_to_html(st.session_state['full_report'])
                st.markdown(html_content, unsafe_allow_html=True)      

                # display_visuals()


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

def display_visuals():
    # Custom CSS to inject into the Streamlit page
    st.markdown(
        """
        <style>
        .dashboard-container {
            border: 2px solid #009688;
            border-radius: 5px;
            background-color: #fafafa;
            padding: 8px; /* Reduced padding */
            margin-bottom: 10px; /* Reduced margin */
            box-shadow: 1px 1px 4px lightgrey;
        }
        .dashboard-title {
            color: #009688;
            margin-bottom: 8px; /* Reduced margin */
            text-align: center;
            font-size: 1.6em; /* Smaller font size */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Create two columns within a container
    with st.container():
        # col1, col2 = st.columns(2)
        with center_column:
            st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
            st.markdown('<p class="dashboard-title">Skills and Proficiencies</p>', unsafe_allow_html=True)
            st.pyplot(plot_skills_and_proficiencies(skills_scores, plot_size=(1, 1)))
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
            st.markdown('<p class="dashboard-title">Job Title Similarity *ESCO*</p>', unsafe_allow_html=True)
            # st.pyplot(plot_job_profile_match(job_profile_match_score, plot_size=(1, 1)))
            st.pyplot(plot_spyder(plot_size=(1, 1)))
            st.markdown('</div>', unsafe_allow_html=True)
            
        
        with right_column:
            st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
            st.markdown('<p class="dashboard-title">Language Match</p>', unsafe_allow_html=True)
            st.pyplot(plot_educational_qualifications(education_match, plot_size=(2, 1)))
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
            st.markdown('<p class="dashboard-title">Educational Qualifications</p>', unsafe_allow_html=True)
            st.pyplot(plot_certifications_and_regulatory_knowledge(certifications_presence, certifications_labels, plot_size=(3, 2)))
            st.markdown('</div>', unsafe_allow_html=True)


# display_visuals()
display_cv_processing()

display_job_offer_processing()

def similarity_bar(score):
    st.write('Similarity:', f'{score:.2f}')
    st.progress(score)


# st.markdown('### Education Details')

# Create two columns for the CV and job offer sections
from vis_streamlit import (display_similarity_bar, display_education_comparison,
                     display_language_comparison, create_matplotlib_chart,
                     create_seaborn_chart, toggle_charts)

# Your Streamlit code to take input and store it in JSON format
json_params = {
    'score': 0.76,
    'cv_education': ["Bachelor of Arts, Communications"],
    'job_offer_education': ["Diplomatura in Marketing"],
    'languages_cv': {'English': '🇬🇧'},
    'languages_job_requirement': {'Catalan': '🏳', 'Spanish': '🇪🇸', 'English': '🇬🇧'},
    'data': {'x': [1, 2, 3, 4], 'y': [10, 11, 12, 13]},
    'show_charts': True
}

def toggle_charts():
    st.session_state['show_charts'] = not st.session_state['show_charts']
    
    
if st.session_state['cv_file'] is not None:
    with center_column:
        # Call the functions and pass the parameters as needed
        display_education_comparison(json_params)
        display_language_comparison(json_params)
        display_similarity_bar(json_params)
        st.markdown("-------------------------------------------------")
        if st.button('Show/Hide Charts'):
            toggle_charts()

        if st.session_state.get('show_charts', False):
            # col1, col2 = st.columns(2)

            plot_size = (1,1)
            #  with st.container:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Job Profile Match")
                st.write("This pie chart shows the percentage match of the candidate's job profile with the job description.")
                # Call the function and use st.pyplot to render the figure
                st.pyplot(plot_job_profile_match(job_profile_match_score, plot_size=plot_size))

            with col2:
                st.subheader("Educational Qualifications")
                st.write("The bar chart indicates whether the candidate has the educational qualifications required for the job.")
                # Call the function and use st.pyplot to render the figure
                st.pyplot(plot_educational_qualifications(education_match, plot_size=plot_size))

            with col1:
                st.subheader("Skills and Proficiencies")
                st.write("The radar chart visualizes the candidate's skill levels across various domains required for the job.")
                # Call the function and use st.pyplot to render the figure
                st.pyplot(plot_skills_and_proficiencies(skills_scores, plot_size=plot_size))

            with col2:
                st.subheader("Certifications and Regulatory Knowledge")
                st.write("This bar chart shows whether the candidate possesses the specific certifications or knowledge areas important for the position.")
                # Call the function and use st.pyplot to render the figure
                st.pyplot(plot_certifications_and_regulatory_knowledge(certifications_presence, certifications_labels, plot_size=plot_size))


if st.session_state['show_charts']:
    display_full_report()












#     # Check the session state to decide whether to show the charts
# if st.session_state.get('show_charts', False):
#     create_matplotlib_chart(json_params)
#     create_seaborn_chart(json_params)
# Column for 'Education for CV'

#     col1, col2 = st.columns(2)

#     with col1:
#         st.markdown('### Education for CV')
#         st.text('Bachelor of Arts,\nCommunications')

#     # Column for 'Required Education by Job Offer'
#     with col2:
#         st.markdown('### Required Education by Job Offer')
#         st.text('Diplomatura')

    
#     # Display the similarity score at the bottom
#     similarity = 0.76
#     similarity_bar(similarity)

#     # Function to display similarity score with styling
#     def similarity_bar(score):
#         # Convert the score to a percentage for display
#         score_percentage = f"{score:.2%}"
#         # Create a progress bar with the score
#         st.progress(score)
#         # Display the score as text below the progress bar
#         st.write(f"Similarity Score: {score_percentage}")

#     # Education data for CV and Job Offer
#     cv_education = [
#         "Bachelor of Arts, Communications",
#         "Master of Science, Data Analytics"
#     ]

#     job_offer_education = [
#         "Diplomatura in Marketing",
#         "Advanced Certification in Public Relations"
#     ]

#     # Split the screen into two columns for CV and Job Offer
#     col1, col2 = st.columns(2)

#     # Column for 'Education for CV'
#     with col1:
#         st.markdown('### Education for CV')
#         # Display each education qualification as a bullet point
#         for education in cv_education:
#             st.markdown(f"- {education}")

#     # Column for 'Required Education by Job Offer'
#     with col2:
#         st.markdown('### Required Education by Job Offer')
#         # Display each required education qualification as a bullet point
#         for education in job_offer_education:
#             st.markdown(f"- {education}")

#     # Display the similarity score at the bottom
#     similarity = 0.76
#     similarity_bar(similarity)




#     # Define the languages and corresponding flag emojis
#     languages_cv = {
#         'English': '🇬🇧'
#     }
#     languages_job_requirement = {
#         'Catalan': '🏳', # There's no specific emoji for the Catalan flag, so we use a placeholder.
#         'Spanish': '🇪🇸',
#         'English': '🇬🇧'
#     }


#     st.markdown("-------------------------------------------------")
#     st.markdown('### Language Requirements Comparison')

#     col1, col2 = st.columns(2)
#     # Fill in the CV column
#     with col1:
#         st.markdown('#### CV Languages')
#         for language, flag in languages_cv.items():
#             st.write(f"{flag} {language}")

#     # Fill in the Job Requirement column
#     with col2:
#         st.markdown('#### Job Requirement Languages')
#         for language, flag in languages_job_requirement.items():
#             st.write(f"{flag} {language}")

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# # Sample data
# df = pd.DataFrame({
#     'x': [1, 2, 3, 4],
#     'y': [10, 11, 12, 13]
# })

# # Function to create a Matplotlib plot
# def create_matplotlib_chart(data):
#     plt.figure(figsize=(6, 4))
#     sns.lineplot(data=data, x='x', y='y')
#     plt.tight_layout()
#     return plt

# # Function to create a Seaborn plot
# def create_seaborn_chart(data):
#     plt.figure(figsize=(6, 4))
#     sns.barplot(data=data, x='x', y='y')
#     plt.tight_layout()
#     return plt


# # Initialize session state
# if 'show_charts' not in st.session_state:
#     st.session_state['show_charts'] = False

# # Define function to toggle the state
# def toggle_charts():
#     st.session_state['show_charts'] = not st.session_state['show_charts']

# with center_column:
#     # Button to toggle charts and report visibility
#     st.button('Show/Hide Charts and Report', on_click=toggle_charts)

#     # If the state is True, display charts and report
#     if st.session_state['show_charts']:
#         # Split the screen into two columns
#         col1, col2 = st.columns(2)
        
#         with col1:
#             # Display Matplotlib chart in the left column
#             st.write("Matplotlib Chart:")
#             plt1 = create_matplotlib_chart(df)
#             st.pyplot(plt1)
            
#             # Display Seaborn chart below the Matplotlib chart
#             st.write("Seaborn Chart:")
#             plt2 = create_seaborn_chart(df)
#             st.pyplot(plt2)
        
#         with col2:
#             # Display the report in the right column
#             st.write("Report:")
#             st.text("Here is some text that represents the report...")
            
#             # You can also use st.markdown for more complex formatting
#             st.markdown("""
#                 ## Report Heading
#                 - Point 1
#                 - Point 2
#                 - Point 3
#             """)





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
# Streamlit code to create a button and run tasks concurrently

# right_column = st.empty()
# center_column = st.empty()
# left_column = st.empty()


#     if st.button('Run Tasks'):
#         with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
#             # Schedule the functions to be executed
#             future_to_function = {
#                 executor.submit(display_cv_processing): "CV Processing",
#                 executor.submit(display_job_offer_processing): "Job Offer Processing",
#                 executor.submit(display_full_report): "Full Report"
#             }
            
#             # Collect the results as they are completed
#             for future in concurrent.futures.as_completed(future_to_function):
#                 function_name = future_to_function[future]
#                 try:
#                     result = future.result()
#                     # Process the result (e.g., display it in the Streamlit app)
#                     st.write(f"{function_name} completed with result: {result}")
#                 except Exception as exc:
#                     st.write(f"{function_name} generated an exception: {exc}")





# # Use the columns to display the charts
# with center_column:
#     with st.container():
#         st.subheader("Job Profile Match")
#         st.pyplot(plot_job_profile_match(job_profile_match_score))
#     with st.container():
#         st.subheader("Skills and Proficiencies")
#         st.pyplot(plot_skills_and_proficiencies(skills_scores))

# with right_column:
#     with st.container():
#         st.subheader("Educational Qualifications")
#         st.pyplot(plot_educational_qualifications(education_match))
#     with st.container():
#         st.subheader("Certifications and Regulatory Knowledge")
#         st.pyplot(plot_certifications_and_regulatory_knowledge(certifications_presence, certifications_labels))

# with center_column:
#     with right_column:
#         with st.container:
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.subheader("Job Profile Match")
#                 st.write("This pie chart shows the percentage match of the candidate's job profile with the job description.")
#                 # Call the function and use st.pyplot to render the figure
#                 st.pyplot(plot_job_profile_match(job_profile_match_score))

#             with col2:
#                 st.subheader("Educational Qualifications")
#                 st.write("The bar chart indicates whether the candidate has the educational qualifications required for the job.")
#                 # Call the function and use st.pyplot to render the figure
#                 st.pyplot(plot_educational_qualifications(education_match))

#             with col1:
#                 st.subheader("Skills and Proficiencies")
#                 st.write("The radar chart visualizes the candidate's skill levels across various domains required for the job.")
#                 # Call the function and use st.pyplot to render the figure
#                 st.pyplot(plot_skills_and_proficiencies(skills_scores))

#             with col2:
#                 st.subheader("Certifications and Regulatory Knowledge")
#                 st.write("This bar chart shows whether the candidate possesses the specific certifications or knowledge areas important for the position.")
#                 # Call the function and use st.pyplot to render the figure
#                 st.pyplot(plot_certifications_and_regulatory_knowledge(certifications_presence, certifications_labels))


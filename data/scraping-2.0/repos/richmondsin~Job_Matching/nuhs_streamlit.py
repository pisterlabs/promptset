# For windows users, please remove | tail -n 1 below
# pip install chromadb==0.3.27 | tail -n 1
# pip install sentence_transformers | tail -n 1
# pip install pandas | tail -n 1
# pip install rouge_score | tail -n 1
# pip install nltk | tail -n 1
# pip install "ibm-watson-machine-learning>=1.0.312" | tail -n 1
# pip install --upgrade ibm_watson_machine_learning

import os
import re
from dotenv import load_dotenv
import pandas as pd
from typing import Optional, Dict, Any, Iterable, List
import pdb
from langchain.document_loaders import PyPDFLoader
# Foundation Models on watsonx
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods
from ibm_watson_machine_learning.foundation_models import Model
import streamlit as st
import logging
from langchain.callbacks import StdOutCallbackHandler
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from PIL import Image
import tempfile

from langChainInterface import LangChainInterface

# Most GENAI logs are at Debug level.
logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))

st.set_page_config(
    page_title="Retrieval Augmented Generation",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.header("Retrieval Augmented Generation with watsonx.ai 💬")

load_dotenv()

handler = StdOutCallbackHandler()

          
try: 
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("Could not import sentence_transformers. Please install sentence-transformers package.")

try:
    import chromadb
    from chromadb.api.types import EmbeddingFunction
except ImportError:
    raise ImportError("Could not import chromdb: Please install chromadb package.")

# watsonx API connection
credentials = {
    "url": os.getenv("IBM_CLOUD_URL", None),
    "apikey": os.getenv("API_KEY", None)
}

# project id
try:
    project_id = os.getenv("PROJECT_ID", None)
except KeyError:
    project_id = input("Please enter your project_id (hit enter): ")

model_id = ModelTypes.LLAMA_2_70B_CHAT

# Greedy method
parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 700,
}

# # Sample method
# parameters = {
#     GenParams.DECODING_METHOD: DecodingMethods.SAMPLE,
#     GenParams.MIN_NEW_TOKENS: 1,
#     GenParams.TOP_K: 20,
#     GenParams.TOP_P: 0.4,
#     GenParams.TEMPERATURE: 0.4,
#     GenParams.MAX_NEW_TOKENS: 500
# }

model = Model(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=project_id
)

# Index knowledge base
# Load data
datasets = ['nuhs']
dataset = datasets[0]    # The current dataset to use
data_root = "data"
data_dir = os.path.join(data_root, dataset)
max_docs = -1
# print("Selected dataset:", dataset)

def load_data_v1(data_dir):
    passages = pd.read_csv(os.path.join(data_dir, "jobs_1.tsv"), sep='\t', header=0)
    return passages

documents = load_data_v1(data_dir)

# Add the education and experience entities to the new columns indextext1 and indextext2 in documents
documents['indextext1'] = documents['id'].astype(str) + "\n" + documents['education'] + "\n" + documents['job'] + "\n" + documents['url']
documents['indextext2'] = documents['id'].astype(str) + "\n" + documents['experience'] + "\n" + documents['job'] + "\n" + documents['url']

# Create embedding function
class MiniLML6V2EmbeddingFunction(EmbeddingFunction):
    MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    def __call__(self, texts):
        return MiniLML6V2EmbeddingFunction.MODEL.encode(texts).tolist()
emb_func = MiniLML6V2EmbeddingFunction()

# Set up Chroma upsert
class ChromaWithUpsert:
    def __init__(self, name,persist_directory, embedding_function,collection_metadata: Optional[Dict] = None,
    ):
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._embedding_function = embedding_function
        self._persist_directory = persist_directory
        self._name = name
        self._collection = self._client.get_or_create_collection(
            name=self._name,
            embedding_function=self._embedding_function
            if self._embedding_function is not None
            else None,
            metadata=collection_metadata,
        )

    def upsert_texts(
        self,
        texts: Iterable[str],
        metadata: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.
        Args:
            :param texts (Iterable[str]): Texts to add to the vectorstore.
            :param metadatas (Optional[List[dict]], optional): Optional list of metadatas.
            :param ids (Optional[List[str]], optional): Optional list of IDs.
            :param metadata: Optional[List[dict]] - optional metadata (such as title, etc.)
        Returns:
            List[str]: List of IDs of the added texts.
        """
        # TODO: Handle the case where the user doesn't provide ids on the Collection
        if ids is None:
            import uuid
            ids = [str(uuid.uuid1()) for _ in texts]
        embeddings = None
        self._collection.upsert(
            metadatas=metadata, documents=texts, ids=ids
        )
        return ids

    def is_empty(self):
        return self._collection.count()==0

    def query(self, query_texts:str, n_results:int=5):
        """
        Returns the closests vector to the question vector
        :param query_texts: the question
        :param n_results: number of results to generate
        :return: the closest result to the given question
        """
        return self._collection.query(query_texts=query_texts, n_results=n_results)

# Embed and index education documents with Chroma
chroma_education = ChromaWithUpsert(
    name=f"{dataset}_minilm6v2_education",
    embedding_function=emb_func,  # you can have something here using /embed endpoint
    persist_directory=data_dir,
)
if chroma_education.is_empty():
    _ = chroma_education.upsert_texts(
        texts=documents.indextext1.tolist(),
        # we handle tokenization, embedding, and indexing automatically. You can skip that and add your own embeddings as well
        metadata=[{'id': id, 'education': education, 'job': job, 'url': url}
                  for (id, education, job, url) in
                  zip(documents.id, documents.education, documents.job, documents.url)],  # filter on these!
        ids=[str(i) for i in documents.id],  # unique for each doc
    )

# Embed and index experience documents with Chroma
chroma_experience = ChromaWithUpsert(
    name=f"{dataset}_minilm6v2_experience",
    embedding_function=emb_func,  # you can have something here using /embed endpoint
    persist_directory=data_dir,
)
if chroma_experience.is_empty():
    _ = chroma_experience.upsert_texts(
        texts=documents.indextext2.tolist(),
        # we handle tokenization, embedding, and indexing automatically. You can skip that and add your own embeddings as well
        metadata=[{'id': id, 'experience': experience, 'job': job, 'url': url}
                  for (id, experience, job, url) in
                  zip(documents.id, documents.experience, documents.job, documents.url)],  # filter on these!
        ids=[str(i) for i in documents.id],  # unique for each doc
    )

# Sidebar contents
with st.sidebar:
    st.title("RAG App")
    st.markdown('''
    ## About
    This app is an LLM-powered RAG built using:
    - [IBM Generative AI SDK](https://github.com/IBM/ibm-generative-ai/)
    - [HuggingFace](https://huggingface.co/)
    - [IBM watsonx.ai](https://www.ibm.com/products/watsonx-ai) LLM model
 
    ''')
    st.write('Powered by [IBM watsonx.ai](https://www.ibm.com/products/watsonx-ai)')
    image = Image.open('watsonxai.jpg')
    st.image(image, caption='Powered by watsonx.ai')
    max_new_tokens= st.number_input('max_new_tokens',1,1024,value=700)
    min_new_tokens= st.number_input('min_new_tokens',0,value=1)
    repetition_penalty = st.number_input('repetition_penalty',1,2,value=1)
    decoding = st.text_input(
            "Decoding",
            "greedy",
            key="placeholder",
        )
    
uploaded_files = st.file_uploader("Choose a PDF file", accept_multiple_files=True)

@st.cache_data
def read_pdf(uploaded_files,
                start_page: int = 1,
                end_page: Optional[int | None] = None) -> list[str]:
    for uploaded_file in uploaded_files:
      bytes_data = uploaded_file.read()
      with tempfile.NamedTemporaryFile(mode='wb', delete=False) as temp_file:
      # Write content to the temporary file
          temp_file.write(bytes_data)
          filepath = temp_file.name
          with st.spinner('Waiting for the file to upload'):
            loader = PyPDFLoader(filepath)
            pages = loader.load()
            total_pages = len(pages)
            if end_page is None:
                end_page = len(pages)
            text_list = []
            for i in range(start_page-1, end_page):
                text = pages[i].page_content
                text = text.replace("\n", " ")  
                text = re.sub(r'\s+', ' ', text)
                text_list.append(text)
            return text_list

if uploaded_files:
    resume = read_pdf(uploaded_files)

    # Convert the resume pages into a single string
    full_resume_text = ' '.join(resume)

    # Keywords for detecting sections
    education_keywords = ["education", "EDUCATION"]
    experience_keywords = ["related experience", "RELATED EXPERIENCE"]

    # Function to extract text between two keywords
    def extract_text_between_keywords(text, start_keyword, end_keywords):
        start_index = text.lower().find(start_keyword.lower())

        # Find the index of the first occurrence of any end keyword after the start index
        end_index = min(
            (text.lower().find(end_keyword.lower(), start_index + len(start_keyword)) for end_keyword in end_keywords if start_index != -1),
            default=len(text)  # Default to the end of the text if no end keyword is found
        )

        if start_index != -1:
            return text[start_index + len(start_keyword):end_index].strip()
        else:
            return None

    # Extract text based on the identified keywords and no specific ending marker for experience
    education_text = None
    experience_text = None

    # Possible ending markers for education section
    education_end_markers = ["honors/awards", "HONORS/AWARDS"]

    for education_keyword in education_keywords:
        education_text = extract_text_between_keywords(full_resume_text, education_keyword, education_end_markers)
        if education_text:
            break

    # For experience section, no specific ending marker mentioned
    experience_end_markers = []

    for experience_keyword in experience_keywords:
        experience_text = extract_text_between_keywords(full_resume_text, experience_keyword, experience_end_markers)
        if experience_text:
            break

    # Query education chunk against the vector database for education
    education_results = chroma_education.query(
        query_texts=[education_text],
        n_results=20,
    )

    # Query education chunk against the vector database for experience
    experience_results = chroma_experience.query(
        query_texts=[experience_text],
        n_results=20,
    )

    # Extract IDs and distances from the results
    education_ids = education_results['ids'][0]
    education_distances = education_results['distances'][0]
    experience_ids = experience_results['ids'][0]
    experience_distances = experience_results['distances'][0]

    # Create dictionaries to map IDs to distances
    education_dict = dict(zip(education_ids, education_distances))
    experience_dict = dict(zip(experience_ids, experience_distances))

    # Define a threshold for filtering job postings
    threshold = 0.68 # out of 1.0 relevance, and 0.70 has 1 only and it is endoscopy

    # Calculate total_distance for all job_ids
    all_total_distances = []
    # Calculate total_distance for all job_ids and include job IDs in separate arrays for education and experience
    all_education_distances = []
    all_experience_distances = []
    for job_id in set(education_ids + experience_ids):  # Use the set to get unique IDs
        education_distance = education_dict.get(job_id)
        experience_distance = experience_dict.get(job_id)

        total_distance = education_distance + experience_distance
        all_total_distances.append({'job_id': job_id, 'total_distance': total_distance})

        all_education_distances.append({'job_id': job_id, 'education_distance': education_distance})
        all_experience_distances.append({'job_id': job_id, 'experience_distance': experience_distance})

    # Sort the lists of dictionaries by education_distance and experience_distance
    all_education_distances.sort(key=lambda x: x['education_distance'])
    all_experience_distances.sort(key=lambda x: x['experience_distance'])

    # Extract the largest education_distance and experience_distance
    largest_education_distance = all_education_distances[-1]['education_distance']
    largest_experience_distance = all_experience_distances[-1]['experience_distance']

    # Filter job postings based on the sum of distance scores and threshold
    filtered_jobs = []
    for job_id in set(education_ids + experience_ids):
        education_distance = education_dict.get(job_id)
        experience_distance = experience_dict.get(job_id)
        
        total_distance = education_distance + experience_distance

        # Calculate relevance score using the provided formula
        # Set weights for education and experience distances
        weight_education = 0.3  # Adjust this based on the importance of education
        weight_experience = 0.7  # Adjust this based on the importance of experience

        # Calculate relevance scores for education and experience distances
        relevance_score_education = 1.0 - ((education_distance / largest_education_distance) * weight_education)
        relevance_score_experience = 1.0 - ((experience_distance / largest_experience_distance) * weight_experience)

        # Calculate relevance score as a weighted average
        relevance_score = (relevance_score_education + relevance_score_experience) / 2.0

        # Ensure the relevance score is within the range [0, 1.0]
        relevance_score = max(0, min(relevance_score, 1.0))

        # Check if the total_distance is below the threshold
        if relevance_score >= threshold:
            filtered_jobs.append({
                'id': job_id,
                'total_distance': total_distance,
                'education_distance': education_distance,
                'experience_distance': experience_distance,
                'relevance_score': relevance_score
            })
    # Sort filtered_jobs based on relevance_score in descending order
    filtered_jobs_sorted = sorted(filtered_jobs, key=lambda x: x['relevance_score'], reverse=True)

    # Combine education and experience chunks into a single context for each row
    combined_contexts = []
    for filtered_job in filtered_jobs_sorted:
        job_id = filtered_job['id']
        
        # Find the corresponding document in the original dataset
        document_row = documents[documents['id'].astype(str) == job_id].iloc[0]
        job = document_row['job']
        education_context = document_row['education']
        experience_context = document_row['experience']
        relevance_score = filtered_job['relevance_score']
        url = document_row['url']

        combined_context = f"Job: {job}Education: {education_context}Experience: {experience_context}Relevance Score: {relevance_score}Url: {url}"
        combined_contexts.append({
            'id': job_id,
            'combined_context': combined_context
        })

# Feed the context and the question to openai model
def make_prompt(contexts, resume, question_text):
    ranked_contexts = "\n\n\n".join(f"Rank {i+1} Context:{context['combined_context']}{'='*20}" for i, context in enumerate(contexts))
    return (
        '''<s>[INST] <<SYS>>
        You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
        If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
        <</SYS>>'''
        + f"Ranked Contexts: {ranked_contexts}"
        + f"Resume: {resume}."
        + f"Question: {question_text}"
        + "[/INST]"
    )

# show user input
if user_question := st.text_input(
    "Question:"
):
    question_text = f'''{user_question} 
    Provide a justification for your answer at the end. List down the jobs according to their Relevance Score in descending order. Answer the qeustion strictly based on the contexts above only, do not use any information from other sources. Answer in the following format:
    Rank:
    Job:
    Url:
    Relevance Score:
    \n
    Justification:
    '''
    # Create the prompt using the extracted contexts if there are any
    if len(combined_contexts) == 0:
        prompt = '''<s>[INST] <<SYS>>
        You are a kind and polite assistant. <</SYS>> 
        Kindly tell the user that there are no relevant jobs found according to the resume that the user uploaded, say it in first person view.[/INST]
        '''
    else:
        prompt = make_prompt(
            contexts=combined_contexts,
            resume=resume[0],  # Assuming you want to use the first page of the resume
            question_text=question_text
        )
    # Generate the answers
    response = model.generate_text(prompt=prompt)

    st.text_area(label="Model Response", value=response, height=700)
    st.write()
# streamlit run nuhs_streamlit.py
# What are the 3 most relevant jobs for this resume? 
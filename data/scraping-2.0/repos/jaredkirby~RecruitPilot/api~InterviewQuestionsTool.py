import os
import io
import logging
import tempfile
import fastapi
import uvicorn
from zipfile import ZipFile

from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile

from langchain.document_loaders import PDFPlumberLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

from api_config import (
    PROMPTS_MAPPING,
    MODEL_QUESTIONS,
)

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ingest_pdf(resume_file_buffer):
    print("Loading resume...")
    try:
        # Create a temporary file manually
        temp_file_path = tempfile.mktemp(suffix=".pdf")
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(resume_file_buffer.read())

        # Use the temporary file path
        loader = PDFPlumberLoader(temp_file_path)
        documents = loader.load()
        resume_text = " ".join(document.page_content for document in documents)
        print("Resume loaded successfully.")

        # Delete the temporary file
        os.remove(temp_file_path)

        return resume_text
    except Exception as e:
        logger.error(f"An error occurred while loading the resume: {e}")
        raise


def get_parameters(selected_job, job_description_input):
    if selected_job == "Input your own":
        job_description = job_description_input
    else:
        selected_prompts = PROMPTS_MAPPING[selected_job]
        job_description = (
            job_description_input
            if job_description_input
            else selected_prompts["job_description"]
        )
    return job_description


# TODO: Switch to OpenAI function LLM call for more reliable response formatting - not an issue for now
def get_questions(
    resume_text,
    job_description,
):
    print("Getting score...")
    llm = ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        temperature=0.0,
        openai_api_key=openai_api_key,
    )

    template = f"""\
You are an Industrial/Organizational Psychologist who is preparing to analyze an applicant based on a job description and resume, 
and create a selection of interview questions specific to the applicant in order to determine their potential success in the role.

Applicant Resume:
-----------------
{resume_text}
-----------------

Job Key Areas of Responsibility:
-----------------
{job_description}
-----------------

Based on the job description and the information provided in the resume, please respond with an analysis of this applicant and a 
selection of interview questions specific to this applicant and designed to understand better if this person will succeed in this role.

Your Response Format:
Applicant Name

List of positive attributes for the position

List of negative attributes for the position

List of questions for the interview
    """

    user_prompt = HumanMessagePromptTemplate.from_template(template=template)
    chat_prompt = ChatPromptTemplate.from_messages([user_prompt])
    formatted_prompt = chat_prompt.format_prompt(
        resume_text=resume_text,
        job_description=job_description,
    ).to_messages()
    # print(formatted_prompt)
    llm = llm
    result = llm(formatted_prompt)
    return result.content


def parse_input(file):
    if file:
        resume_file_buffer = io.BytesIO(file.getbuffer())
        return ingest_pdf(resume_file_buffer)


def select_job():
    if "selected_job" not in st.session_state:
        st.session_state.selected_job = "CEMM - Senior CPG Account Strategist"
    st.session_state.selected_job = st.selectbox(
        "Select an open position or add your own",
        (
            "CEMM - Senior CPG Account Strategist",
            "CEMM - Advertising Assistant",
            "Input your own",
        ),
        index=0,
        key="job_selection",
    )
    return st.session_state.selected_job


selected_job = select_job()

uploaded_resumes = st.file_uploader(
    "Upload Resumes (PDF files)", type=["pdf"], accept_multiple_files=True
)

start_button = st.button("Generate Questions")

if uploaded_resumes and start_button:
    try:
        zip_data, categorization_results = process_resumes(uploaded_resumes)
        if zip_data:
            st.download_button(
                label="✨ Download Scores ✨",
                data=zip_data,
                file_name="scores.zip",
                mime="application/zip",
            )
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        logger.error(f"An error occurred during processing: {e}")


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    resume_file_buffer = io.BytesIO(file.file.read())
    resume_text = ingest_pdf(resume_file_buffer)
    return {"filename": file.filename}


@app.post("/generate_questions/")
async def generate_questions(
    resume_text: str, selected_job: str, job_description_input: str = None
):
    job_description = get_parameters(selected_job, job_description_input)
    questions = get_questions(resume_text, job_description)
    return {"questions": questions}

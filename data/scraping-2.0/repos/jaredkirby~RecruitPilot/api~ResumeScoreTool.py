import os
import io
import logging
import tempfile
import streamlit as st
from zipfile import ZipFile

from langchain.document_loaders import PDFPlumberLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

from config import (
    PAGE_TITLE,
    PAGE_ICON,
    SUB_TITLE,
    LAYOUT,
    PROMPTS_MAPPING,
    MODEL,
)

openai_api_key = st.secrets["OPENAI_API_KEY"]

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_data
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
        st.error(f"An error occurred while loading the resume: {e}")
        raise


def save_to_category_buffer(
    category, applicant_name, resume_bytes, response_content, main_zip
):
    # Add the PDF file
    main_zip.writestr(f"{category}/{applicant_name}/{applicant_name}.pdf", resume_bytes)

    # Add the response text
    main_zip.writestr(
        f"{category}/{applicant_name}/{applicant_name}_response.txt",
        response_content,
    )


def get_parameters(
    selected_job, job_description_input, high_fit_resume_input, low_fit_resume_input
):
    if selected_job == "Input your own":
        job_description = job_description_input
        high_fit_resume = high_fit_resume_input
        low_fit_resume = low_fit_resume_input
    else:
        selected_prompts = PROMPTS_MAPPING[selected_job]
        job_description = (
            job_description_input
            if job_description_input
            else selected_prompts["job_description"]
        )
        high_fit_resume = (
            high_fit_resume_input
            if high_fit_resume_input
            else selected_prompts["high_fit_resume"]
        )
        low_fit_resume = (
            low_fit_resume_input
            if low_fit_resume_input
            else selected_prompts["low_fit_resume"]
        )
    return job_description, high_fit_resume, low_fit_resume


# TODO: Switch to OpenAI function LLM call for more reliable response formatting - not an issue for now
@st.cache_data
def get_score(
    resume_text,
    job_description,
    high_fit_resume,
    low_fit_resume,
):
    print("Getting score...")
    llm = ChatOpenAI(
        model=MODEL,
        temperature=0.0,
        openai_api_key=openai_api_key,
    )
    # Step 1: Check for high fit resume
    if high_fit_resume:
        example_high_fit = (
            "Example 'high-fit' resume with a score of 0.99 for reference:"
        )
        h_div = "-----------------"
    else:
        example_high_fit = ""
        h_div = ""
        high_fit_resume = ""

    # Step 2: Check for low fit resume
    if low_fit_resume:
        example_low_fit = "Example 'low-fit' resume with a score of 0.10 for reference:"
        l_div = "-----------------"
    else:
        example_low_fit = ""
        l_div = ""
        low_fit_resume = ""

    template = f"""\
You are an Industrial-Organizational Psychologist who specializes in personnel selection and assessment. 
Your discipline of study, Industrial-Organizational Psychology, would best prepare you to answer the 
question or perform the task of determining a job fit score based on a resume and a job description. 

You will review the following resume and job description and determine a job fit score as a float between 0 and 1 (Example: 0.75) and a short explanation for the score.

Applicant Resume:
-----------------
{resume_text}
-----------------

Job Key Areas of Responsibility:
-----------------
{job_description}
-----------------

{example_high_fit}
{h_div}
{high_fit_resume}
{h_div}

{example_low_fit}
{l_div}
{low_fit_resume}
{l_div}

Remember, your task is to determine a job fit score as a float between 0 and 1 (Example: 0.99) and a short explanation for score.
Respond with only the score and explanation. Do not include the resume or job description in your response.

RESPONSE FORMAT:
Job Fit Score: 
Explanation:

Job Fit Score:
    """

    user_prompt = HumanMessagePromptTemplate.from_template(template=template)
    chat_prompt = ChatPromptTemplate.from_messages([user_prompt])
    formatted_prompt = chat_prompt.format_prompt(
        resume_text=resume_text,
        job_description=job_description,
        high_fit_resume=high_fit_resume,
        low_fit_resume=low_fit_resume,
        l_div=l_div,
        h_div=h_div,
    ).to_messages()
    # print(formatted_prompt)
    llm = llm
    result = llm(formatted_prompt)
    return result.content


def parse_score_and_explanation(result_content):
    # Assuming the score and explanation are on separate lines
    lines = result_content.split("\n")
    score = float(lines[0])  # Assuming the score is on the first line
    explanation = lines[1] if len(lines) > 1 else ""  # Explanation on the second line
    return score, explanation


def categorize_score(score, threshold1, threshold2):
    if score > threshold1:
        return "best"
    elif score > threshold2:
        return "good"
    else:
        return "rest"


def parse_resume_bytes(resume_bytes):
    resume_file_buffer = io.BytesIO(resume_bytes)
    resume_text = ingest_pdf(resume_file_buffer)
    return resume_text


def parse_input(file, text_input_key):
    if file:
        resume_file_buffer = io.BytesIO(file.getbuffer())
        return ingest_pdf(resume_file_buffer)
    else:
        return st.session_state[text_input_key]


def process_resumes(uploaded_resumes):
    # Define the input variables here based on the selected job
    job_description_input = st.session_state.job_description_input
    high_fit_resume_input = st.session_state.high_fit_resume_input
    low_fit_resume_input = st.session_state.low_fit_resume_input

    # Create a dictionary to store the categorization results
    categorization_results = {"best": [], "good": [], "rest": []}

    # Parse the custom user input
    job_description = parse_input(job_description_file, "job_description_input")
    high_fit_resume = parse_input(high_fit_resume_file, "high_fit_resume_input")
    low_fit_resume = parse_input(low_fit_resume_file, "low_fit_resume_input")

    # Get the values for job_description, high_fit_resume, and low_fit_resume
    job_description, high_fit_resume, low_fit_resume = get_parameters(
        selected_job, job_description_input, high_fit_resume_input, low_fit_resume_input
    )

    zip_buffer = io.BytesIO()
    try:
        with ZipFile(zip_buffer, "a") as main_zip:
            for i, resume_file in enumerate(uploaded_resumes):
                # User stopping mechanism
                if st.session_state.stop_button_clicked:
                    st.session_state.status_text = "Process stopped by user."
                    break

                # Update progress
                st.session_state.status_text = (
                    f"Processing resume {i + 1}/{len(uploaded_resumes)}..."
                )

                # Read and parse resume bytes
                resume_bytes = resume_file.getbuffer()
                resume_text = parse_resume_bytes(resume_bytes)

                # Get score for the resume
                result_content = get_score(
                    resume_text, job_description, high_fit_resume, low_fit_resume
                )
                score, _ = parse_score_and_explanation(result_content)

                # Categorize the score
                category = categorize_score(score, best_select, good_select)

                # Save the resume to the appropriate category buffer
                applicant_name = os.path.splitext(resume_file.name)[0]
                categorization_results[category].append(applicant_name)

                save_to_category_buffer(
                    category, applicant_name, resume_bytes, result_content, main_zip
                )

                # Update progress bar
                st.session_state.progress = (i + 1) / len(uploaded_resumes)
                progress_bar.progress(int(st.session_state.progress * 100))

        return zip_buffer.getvalue(), categorization_results

    except Exception as e:
        st.error(f"An error occurred: {e}")
        logger.error(f"An error occurred: {e}")


# Streamlit interface
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout=LAYOUT)

st.markdown(
    f"<h1 style='text-align: center;'>{PAGE_TITLE} {PAGE_ICON} <br> {SUB_TITLE}</h1>",
    unsafe_allow_html=True,
)

st.divider()

# Initialize session state variables
if "stop_button_clicked" not in st.session_state:
    st.session_state.stop_button_clicked = False
if "status_text" not in st.session_state:
    st.session_state.status_text = ""
if "progress" not in st.session_state:
    st.session_state.progress = 0
if "processing" not in st.session_state:
    st.session_state.processing = False
if "job_description_input" not in st.session_state:
    st.session_state.job_description_input = ""
if "high_fit_resume_input" not in st.session_state:
    st.session_state.high_fit_resume_input = ""
if "low_fit_resume_input" not in st.session_state:
    st.session_state.low_fit_resume_input = ""


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

# Initialize the variables outside the conditional block
job_description_file = None
high_fit_resume_file = None
low_fit_resume_file = None

if selected_job == "Input your own":
    with st.expander("Custom Input", expanded=True):
        st.text_area(
            "Job Description Text",
            placeholder="""Summary of Position:
This in-house position is crucial for supporting our client's ...
            """,
            key="job_description_input",
        )
        job_description_file = st.file_uploader("Or upload a PDF", type=["pdf"])

        have_resume_examples = st.checkbox(
            "I have good and/or bad resume examples",
            help="Adding examples of good and bad resumes can increase scoring accuracy",
            key="have_resume",
        )
        if have_resume_examples:
            high_fit_resume_file = st.file_uploader(
                "Upload High-Fit Resume Example (optional)", type=["pdf"]
            )
            is_high_text = st.checkbox("Enter as text", key="is_high_text")
            if is_high_text:
                st.text_area("High-Fit Text", key="high_fit_resume_input")

            low_fit_resume_file = st.file_uploader(
                "Upload Low-Fit Resume Example (optional)", type=["pdf"]
            )
            is_low_text = st.checkbox("Enter as text", key="is_low_text")
            if is_low_text:
                st.text_area("Low-Fit Text)", key="low_fit_resume_input")


uploaded_resumes = st.file_uploader(
    "Upload Resumes (PDF files)", type=["pdf"], accept_multiple_files=True
)

if uploaded_resumes:
    best_select = st.slider(
        "Select a 'best' score threshold",
        0.0,
        1.0,
        0.8,
        help="Default is 0.8. The lower the threshold, the more resumes will be categorized as 'best'.",
    )
    good_select = st.slider(
        "Select a 'good' score threshold",
        0.0,
        1.0,
        0.6,
        help="Default is 0.6. The lower the threshold, the more resumes will be categorized as 'good'.",
    )


start_button = st.button("Start Scoring Resumes")


status_text = st.empty()
status_text.text(st.session_state.status_text)  # Status text now updates dynamically

if start_button:
    st.session_state.stop_button_clicked = False
    st.session_state.processing = True
    st.session_state.progress = 0


# Display the stop button only when processing is True
if st.session_state.processing:
    # Update progress bar and status text from session state
    progress_bar = st.progress(0)
    progress_bar.progress(int(st.session_state.progress * 100))
#    if uploaded_resumes:
#        stop_button = st.button("Stop Process")
#        if stop_button:
#            st.session_state.stop_button_clicked = True
#            st.session_state.status_text = "Stopping the process..."
#            st.session_state.processing = False

if uploaded_resumes and start_button:
    st.session_state.status_text = "Starting the process..."
    try:
        zip_data, categorization_results = process_resumes(uploaded_resumes)
        if zip_data:
            st.download_button(
                label="✨ Download Scores ✨",
                data=zip_data,
                file_name="scores.zip",
                mime="application/zip",
            )
            # Displaying the results
            st.markdown("##### Categorized:")
            for category, resumes in categorization_results.items():
                st.write(f"{category.capitalize()}: {len(resumes)}")

            st.markdown("##### Your 'best' applicants:")
            st.write(", ".join(categorization_results["best"]))
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        logger.error(f"An error occurred during processing: {e}")
    st.session_state.status_text = "Process completed. Download the scores below."
    st.session_state.processing = False
else:
    if start_button:
        st.warning("Please upload resumes before starting the process.")


with st.expander("🤔 How to Use"):
    st.info(
        f"""
This tool 🛠️ uses the OpenAI API 🧠 to score (with an explanation) and categorize resumes based on a job description Then provides a downloadable file with the original resumes in their respective categories.

1️⃣) Select a job description from the dropdown menu or input your own ✏️.

1️⃣ .5) If you input your own job description, you can also upload a PDF or enter text examples of a high-fit ✅ and low-fit ❌ resume to the specified job description. This will increase the accuracy 🎯 of the scoring.

2️⃣) Upload resumes to score and categorize 📑. You can upload multiple resumes at once.

3️⃣) Click the "Start Scoring Resumes" button.

4️⃣) Once the process is complete, a download button will appear to export the scores and categorized resumes 📥.

📌Note: The OpenAI models do make mistakes 😅 and the results may not be perfect ✨. If you have any questions or feedback, please reach out to me on [Twitter](https://twitter.com/Kirby_).
            """
    )

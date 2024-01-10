import streamlit as st
from openai import OpenAI
import requests
from langchain.document_loaders import PyPDFLoader
import os

ID_jobs = {
        "Software Engineer".lower() : "SWE",
        "Project Manager".lower() : "PJM",
        "Product Manager".lower() : "PDM",
        "Business Analyst".lower() : "BSA",
        "Customer Success Manager".lower() : "CSM",
        "Data Analyst".lower() : "DAN",
        "Data Scientist".lower() : "DSC",
        "Account Executive".lower() : "ACE",
        "Data Engineer".lower() : "DEN",
        "Sales Representative".lower() : "SRS",
        "Quality Engineer".lower() : "QEN",
        "Solutions Engineer".lower() : "SEN",
        "Marketing Manager".lower() : "MKT",
        "DevOps Engineer".lower() : "DEV",
        "System Engineer".lower() : "SYS",
    }

job_images = {
    "software engineer": './app/public/swe.gif',
    "project manager": './app/public/pjm.jpg',
    "product manager": './app/public/pdm.png',
    "business analyst": './app/public/bsa.jpg',
    "customer success manager": './app/public/csm.png',
    "data analyst": './app/public/dan.png',
    "data scientist": './app/public/dsc.png',
    "account executive": './app/public/ace.jpg',
    "data engineer": './app/public/den.jpg',
    "sales representative": './app/public/srs.jpg',
    "quality engineer": './app/public/qen.png',
    "solutions engineer": './app/public/sen.png',
    "marketing manager": './app/public/mkt.png',
    "devops engineer": './app/public/dev.jpg',
    "system engineer": './app/public/sys.jpg',
}

st.set_page_config(page_title="🔎 Resume Analysis")

st.title("CV Analyser")

uploaded_file = st.file_uploader(
    label="Upload your CV",
    type=['pdf'])

open_ai_key = st.text_input(
    label="OpenAI Key - To extract the skill of your CV"
)
st.markdown(
    """<style>
    div[data-testid="stFileUploader"] > label > div[data-testid="stMarkdownContainer"] > p {
    font-size: 20px;
}
    </style>
    """, unsafe_allow_html=True)

submit_button = st.button("Extract skill and predict 🔮")

def analyse(uploaded_file, oai_key) -> str:

    if uploaded_file is not None:
    # Define the file path (you can choose a specific directory)
        file_path = f"./temp_files/{uploaded_file.name}"

        # Create a directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Write the uploaded file to the new file path
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()

    # Now you can access the API key
    prompt = pages[0].page_content
    client = OpenAI(
        api_key=oai_key
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content":
                "You are a classic recruiter, you only rely on fact, you don't imagine anything, Give me all the skills of the person, only answer in english. Make it concise 5 lines max."},
            {"role": "user", "content": prompt}
        ]
    )

    try:
        os.remove(file_path)
        #st.info("Temporary file deleted.")
    except OSError as e:
        st.error(f"Error: {e.strerror}")

    return response.choices[0].message.content

if submit_button:
    if st.secrets["OpenAI_API_KEY"]:
        oai_key = st.secrets["OpenAI_API_KEY"]
    else:
        oai_key = open_ai_key

    query = analyse(uploaded_file, oai_key)
    st.write(query)
    if query:
        # Process the inputs
        url = f'{st.secrets["api_url"]}/predict'
        payload = {
            'type': 'dense',
            'query': query
        }
        r = requests.post(url, params=payload)

        response = r.json()
        prefix = "You would be a phenomenal "

        # Remove the prefix from the string
        job_title = list(response)[0][len(prefix):].lower()
        #job_title = list(response)[0].lower()

        if job_title in job_images:
            st.header(job_title.capitalize())
            st.image(job_images[job_title])
        else:
            st.header("Error 404, No Job Found 🥲")
        # job_id = response['top_match']['id']

        # if job_id == "SWE":
        #     st.header(response['top_match']['title'])
        #     st.image('./app/public/swe.jpg')

        # elif job_id == "PJM":
        #     st.header(response['top_match']['title'].capitalize())
        #     st.image('./app/public/pjm.jpg')

        # elif job_id == "PDM":
        #     st.header(response['top_match']['title'])
        #     st.image('./app/public/pdm.jpg')

        # elif job_id == "BSA":
        #     st.header(response['top_match']['title'])
        #     st.image('./app/public/bsa.jpg')

        # elif job_id == "CSM":
        #     st.header(response['top_match']['title'])
        #     st.image('./app/public/csm.jpg')

        # elif job_id == "DAN":
        #     st.header(response['top_match']['title'])
        #     st.image('./app/public/dan.jpg')

        # elif job_id == "DSC":
        #     st.header(response['top_match']['title'])
        #     st.image('./app/public/job_fair.jpg')

        # elif job_id == "ACE":
        #     st.header(response['top_match']['title'])
        #     st.image('./app/public/job_fair.jpg')

        # elif job_id == "DEN":
        #     st.header(response['top_match']['title'])
        #     st.image('./app/public/job_fair.jpg')

        # elif job_id == "SRS":
        #     st.header(response['top_match']['title'])
        #     st.image('./app/public/job_fair.jpg')

        # elif job_id == "QEN":
        #     st.header(response['top_match']['title'])
        #     st.image('./app/public/job_fair.jpg')

        # elif job_id == "SEN":
        #     st.header(response['top_match']['title'])
        #     st.image('./app/public/job_fair.jpg')

        # elif job_id == "MKT":
        #     st.header(response['top_match']['title'])
        #     st.image('./app/public/job_fair.jpg')

        # elif job_id == "DEV":
        #     st.header(response['top_match']['title'])
        #     st.image('./app/public/job_fair.jpg')

        # elif job_id == "SYS":
        #     st.header(response['top_match']['title'])
        #     st.image('./app/public/sys.jpg')

        # else :
        #     st.header('Error 404, No Job Found 🥲')

    with st.expander("Learn More"):
        st.write(response)

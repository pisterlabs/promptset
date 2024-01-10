import streamlit as st
import pandas as pd
import langchain as lc
from langchain import PromptTemplate
from langchain.llms.bedrock import Bedrock
import os
import sys
import time
import boto3
from pprint import pprint
from streamlit_extras.echo_expander import echo_expander
from streamlit_extras.add_vertical_space import add_vertical_space
import json
from pathlib import Path
from st_pages import show_pages_from_config
from components.utils import display_cover_with_title, reset_session_state
import components.authenticate as authenticate  # noqa: E402
import components.genai_api as genai_api  # noqa: E402
import components.pinpoint_api as pinpoint_api
import s3fs
from components.utils_models import BEDROCK_MODELS

import logging
from streamlit_extras.switch_page_button import switch_page

LOGGER = logging.Logger("AI-Chat", level=logging.DEBUG)
HANDLER = logging.StreamHandler(sys.stdout)
HANDLER.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))
LOGGER.addHandler(HANDLER)

path = Path(os.path.dirname(__file__))
sys.path.append(str(path.parent.parent.absolute()))

#########################
#     COVER & CONFIG
#########################

# titles
COVER_IMAGE = os.environ.get("COVER_IMAGE_URL")
TITLE = "Marketing Content Generator"
DESCRIPTION = "Generate marketing content for your customers"
PAGE_TITLE = "GenAI Marketing Content Generator"
PAGE_ICON = "🧙🏻‍♀️"

# page config
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="centered",
    initial_sidebar_state="expanded",
)

# display cover immediately so that it does not pop in and out on every page refresh
cover_placeholder = st.empty()
with cover_placeholder:
    display_cover_with_title(
        title=TITLE,
        description=DESCRIPTION,
        image_url=COVER_IMAGE,
    )

# custom page names in the sidebar
show_pages_from_config()


#########################
#  CHECK LOGIN (do not delete)
#########################

# switch to home page if not authenticated
authenticate.set_st_state_vars()
if not st.session_state["authenticated"]:
    switch_page("Home")


#########################
#       CONSTANTS
#########################

# page name for caching
PAGE_NAME = "choose_segment"
# Poll interview to get export job status
POLL_INTERVAL = 1

# default model specs
with open(f"{path.parent.absolute()}/components/model_specs.json") as f:
    MODEL_SPECS = json.load(f)

# Hardcoded lists of available and non available models.
# If you want to add new available models make sure to update those lists as well as model_specs dict
MODELS_DISPLAYED = BEDROCK_MODELS
MODELS_UNAVAILABLE = [
    "LLAMA 2",
    "Falcon",
    "Flan T5",
]  # Models that are not available for deployment
MODELS_NOT_DEPLOYED = []  # Remove models from this list after deploying the models

# Map each job status to a progress value (percentage)
JOB_STATUS_PROGRESS_MAP = {
    "CREATED": 10,
    "PREPARING_FOR_INITIALIZATION": 20,
    "INITIALIZING": 30,
    "PROCESSING": 40,
    "PENDING_JOB": 50,
    "COMPLETING": 60,
    "COMPLETED": 100,
    "FAILING": 90,
    "FAILED": 0,
}

# Initialize s3fs object
fs = s3fs.S3FileSystem(anon=False)

#########################
# SESSION STATE VARIABLES
#########################

reset_session_state(page_name=PAGE_NAME)
st.session_state.setdefault("ai_model", MODELS_DISPLAYED[0])  # default model
# if "ai_model" not in st.session_state:
#     st.session_state["ai_model"] = MODELS_DISPLAYED[0]  # default model
LOGGER.log(logging.DEBUG, (f"ai_model selected: {st.session_state['ai_model']}"))

# Initialize df in session state if it's not already present
if "df_name" not in st.session_state:
    st.session_state["df_name"] = None

########################################################################################################################################################################
######################################################## Session States and CSS      ###################################################################################
########################################################################################################################################################################

# Define the style of the box element
box_style = {
    "border": "1px solid #ccc",
    "padding": "10px",
    "border-radius": "5px",
    "margin": "10px",
}

########################################################################################################################################################################
######################################################## NavBar      ###################################################################################################
########################################################################################################################################################################


########################################################################################################################################################################
######################################################## Functions    ##################################################################################################
########################################################################################################################################################################


def get_pinpoint_segments():
    with st.spinner("Processing..."):
        segments = pinpoint_api.invoke_pinpoint_segment(
            access_token=st.session_state["access_token"],
        )
        return segments


def create_pinpoint_export_job(segment_id):
    job_response = pinpoint_api.invoke_pinpoint_create_export_job(
        access_token=st.session_state["access_token"], segment_id=segment_id
    )
    return job_response


def get_pinpoint_job_status(job_id):
    job_status_response = pinpoint_api.invoke_pinpoint_export_job_status(
        access_token=st.session_state["access_token"], job_id=job_id
    )
    return job_status_response


def get_export_files_uri(s3_url_prefix, total_pieces):
    job_status_response = pinpoint_api.invoke_s3_fetch_files(
        access_token=st.session_state["access_token"],
        s3_url_prefix=s3_url_prefix,
        total_pieces=total_pieces,
    )
    return job_status_response


def read_s3_file(file_path):
    """Read a gzipped file from S3 and return its content as a DataFrame."""
    with fs.open(file_path, "rb") as f:
        df = pd.read_json(f, compression="gzip", lines=True)
        normalized_df = pd.json_normalize(df.to_dict(orient="records"))
    return normalized_df


def save_df_session_state(df, df_name):
    st.session_state["df"] = df
    st.session_state["df_name"] = df_name


def disable(b):
    st.session_state["disabled"] = b


#########################
#       STYLING
#########################

st.markdown(
    """
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 14px 20px;
        margin: 8px 0;
        border: none;
        cursor: pointer;
        width: 100%;
        opacity: 0.9;
    }
    .stButton>button:hover {
        opacity: 1;
        color: white;
    }
    .stSelectbox>div>div>input {
        pointer-events: none;
    }
</style>
""",
    unsafe_allow_html=True,
)

########################################################################################################################################################################
######################################################## Page Display    ##################################################################################################
########################################################################################################################################################################

st.sidebar.markdown(f"<div style='{box_style}'>", unsafe_allow_html=True)
st.sidebar.markdown(
    f"<p><strong>Current Segment: </strong> {st.session_state['df_name']}</p>",
    unsafe_allow_html=True,
)

st.markdown("## Segment Data in Amazon Pinpoint")

segments = json.loads(get_pinpoint_segments())

if not segments:
    st.error("No Segment Found in Amazon Pinpoint. Please upload a Segment first.")
    st.stop()

# Normalize the JSON into a DataFrame
df = pd.json_normalize(segments)

# Extract the required columns with default value 0 if not present
df["SMS"] = df.get("ImportDefinition.ChannelCounts.SMS", 0)
df["VOICE"] = df.get("ImportDefinition.ChannelCounts.VOICE", 0)
df["EMAIL"] = df.get("ImportDefinition.ChannelCounts.EMAIL", 0)
df["PUSH"] = df.get("ImportDefinition.ChannelCounts.PUSH", 0)

# Select only the required columns
df = df[
    [
        "Name",
        "SegmentType",
        "ImportDefinition.Size",
        "SMS",
        "VOICE",
        "EMAIL",
        "PUSH",
        "Id",
    ]
]

# Rename the columns
df.columns = ["Name", "Type", "Size", "SMS", "VOICE", "EMAIL", "PUSH", "Segment ID"]


st.dataframe(df, hide_index=True)

# Get segment names
segment_names = df["Name"].tolist()

# Create a select box
selected_segment_name = st.selectbox(
    label="Select an Amazon Pinpoint Segment to deep dive:", options=segment_names
)

# Get the selected segment
selected_segment = df[df["Name"] == selected_segment_name]

# Get the Segment ID corresponding to the selected segment name
selected_segment_id = selected_segment["Segment ID"].values[0]

# Create a button placeholder
placeholder = st.empty()
segment_button = placeholder.button(
    "Export and Analyze Segment", disabled=False, key="1"
)

job_status = "FAILED"
status_placeholder = st.empty()
exported_files = None
if segment_button:
    segment_button_2 = placeholder.button(
        "Please wait while we fetch your Amazon Pinpoint Segment",
        disabled=True,
        key="2",
    )
    # Prompt Pinpoint to Export Segment to S3
    get_job_response = json.loads(create_pinpoint_export_job(selected_segment_id))
    job_id = get_job_response["Id"]
    # Keep polling job id until status is completed
    while True:
        # Call your backend server to get the job status
        get_job_status = json.loads(get_pinpoint_job_status(job_id))
        job_status = get_job_status["JobStatus"]
        # Display the current job status as a label
        status_placeholder.text(f"Current Job Status: {job_status}")
        if job_status == "COMPLETED":
            # Export now completed, get the files from S3
            exported_files = json.loads(
                get_export_files_uri(
                    get_job_status["Definition"]["S3UrlPrefix"],
                    get_job_status["TotalPieces"],
                )
            )
            break
        elif job_status == "FAILED":
            st.error(f"Segment {selected_segment_id} export failed.")
            break
        time.sleep(POLL_INTERVAL)

if exported_files is not None:
    empty_button = placeholder.empty()
    status_placeholder.success(
        f"Segment {selected_segment_name} has been successfully retrieved."
    )
    # Create an empty DataFrame
    combined_df = pd.DataFrame()

    # Iterate through the S3 URIs, read each gzipped file, and append to the combined DataFrame
    for file_path in exported_files:
        df = read_s3_file(file_path)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    # Display the combined DataFrame in Streamlit
    st.markdown(f"## Preview of {selected_segment_name} (100 entries)")
    st.dataframe(combined_df.head(100), hide_index=True)
    # Check if the DataFrame is already confirmed and stored in session state
    # Create a button for confirmation
    st.button(
        "Confirm to use this Segment Data",
        on_click=save_df_session_state(
            df=combined_df, df_name=f"(Pinpoint)-{selected_segment_name}"
        ),
    )

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
import components.personalize_api as personalize_api
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
POLL_INTERVAL = 5

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

BUCKET_NAME = os.environ.get("BUCKET_NAME")

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

# Initialize job_name in session state if it's not already present
if "job_name" not in st.session_state:
    st.session_state["job_name"] = None

# Check if 'df_personalize_jobs' is already in the session state
if "df_personalize_jobs" not in st.session_state:
    st.session_state.df_personalize_jobs = pd.DataFrame()


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


def create_personalize_batch_segment(item_ids, num_results):
    personalize_batch_segment_response = (
        personalize_api.invoke_personalize_batch_segment(
            access_token=st.session_state["access_token"],
            item_ids=item_ids,
            num_results=num_results,
        )
    )
    return personalize_batch_segment_response


def get_personalize_job(job_arn):
    job_status_response = personalize_api.invoke_personalize_describe_job(
        access_token=st.session_state["access_token"], job_arn=job_arn
    )
    return job_status_response


def get_personalize_jobs():
    job_status_response = personalize_api.invoke_personalize_get_jobs(
        access_token=st.session_state["access_token"],
    )
    return job_status_response


@st.cache_data(ttl=30)
def cached_get_personalize_jobs():
    return get_personalize_jobs()


def read_s3_file(file_path):
    """Read a JSON file from S3 and return its content."""
    with fs.open(file_path, "rb") as f:
        file_content = f.read().decode("utf-8")  # Read the content and decode it
    return file_content


def process_json_content(content):
    """Process the JSON content and return a DataFrame."""
    # Split the content by newline to get individual JSON strings
    json_strings = content.strip().split("\n")

    # Process each JSON string
    data = []
    for json_str in json_strings:
        item = json.loads(json_str)
        item_id = item["input"]["itemId"]
        for user_id in item["output"]["usersList"]:
            data.append({"itemId": item_id, "userId": user_id})

    # Convert the data to a DataFrame
    df = pd.DataFrame(data)
    return df


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
st.markdown("## Batch Segmentation with Amazon Personalize")

#########################
#       UI To create Batch Segment
#########################
st.markdown("### Generate Recommended Segment")
# Text boxes for user input
num_results = st.number_input("Enter Number of Results", min_value=1, value=3)

# Fetch item metadata
# Get Item Metadata
with fs.open(f"s3://{BUCKET_NAME}/demo-data/df_item_deduplicated.csv", "rb") as f:
    item_data = pd.read_csv(f)
# Sidebar title
st.sidebar.title("Filters")

# Initialize a dictionary to hold selected filter values
filters = {}

# Dynamically create filters for all columns except 'ITEM_ID'
for col in item_data.columns:
    if col != "ITEM_ID":
        # For categorical columns, use a selectbox
        if item_data[col].dtype == "object":
            filters[col] = st.sidebar.selectbox(
                f"Select {col}", options=["All"] + list(item_data[col].unique())
            )
        # For numerical columns, use a slider
        else:
            filters[col] = st.sidebar.slider(
                f"{col} Range",
                float(item_data[col].min()),
                float(item_data[col].max()),
                (float(item_data[col].min()), float(item_data[col].max())),
            )

# Filtering data
filtered_data = item_data.copy()

for col, value in filters.items():
    if value != "All":
        if isinstance(value, tuple):  # For numerical columns
            filtered_data = filtered_data[
                (filtered_data[col] >= value[0]) & (filtered_data[col] <= value[1])
            ]
        else:  # For categorical columns
            filtered_data = filtered_data[filtered_data[col] == value]

st.markdown("#### Filtered Item Dataframe")
# Display filtered DataFrame
st.dataframe(filtered_data)

# Button to trigger the API call
if st.button("Create Batch Segment"):
    # Extract item_ids from the filtered DataFrame and convert them to string
    item_ids = filtered_data["ITEM_ID"].astype(str).tolist()

    # Join the list into a comma-separated string
    item_ids = ",".join(item_ids)

    personalize_batch_segment_response = create_personalize_batch_segment(
        item_ids, num_results
    )
    personalize_batch_segment_response = json.loads(
        personalize_batch_segment_response.decode("utf-8")
    )
    st.session_state["job_name"] = personalize_batch_segment_response[
        "batchSegmentJobArn"
    ].split("/")[-1]
    st.write(
        f'Batch Segment:{st.session_state["job_name"]} for items: {str(item_ids)} Created Successfully!'
    )

st.divider()

#########################
#       GET SPECIFIC INFO ABOUT PERSONALIZE JOB
#########################

try:
    # Run Fetch all segment jobs one time first
    personalize_jobs = cached_get_personalize_jobs()
    # Decode the byte string and parse as JSON
    data = json.loads(personalize_jobs.decode("utf-8"))
    st.session_state.df_personalize_jobs = pd.DataFrame(data["batchSegmentJobs"])
    st.session_state.df_personalize_jobs = (
        st.session_state.df_personalize_jobs.sort_values(
            by="creationDateTime", ascending=False
        )
    )
    show_segment_info = True  # Flag to control UI display
except KeyError:
    st.error("No Segment Job found. Create a segment job first.")
    show_segment_info = False  # Flag to control UI display

# Then, conditionally display the UI based on the flag
if show_segment_info:
    # Your existing code to display the UI for "GET SPECIFIC INFO ABOUT PERSONALIZE JOB"
    st.markdown("### Choose Personalized Segment")
    st.markdown(
        "Once your Segment Job is ACTIVE, choose the Job in the below drop down to view recommended customer information."
    )
    # Create a dropdown for the jobName column
    selected_job_name = st.selectbox(
        label="Select a Personalize Job to view:",
        options=st.session_state.df_personalize_jobs["jobName"].tolist(),
    )

    # Create a button to view the selected segment
    if st.button("View Segment"):
        # Get the selected job
        selected_job = st.session_state.df_personalize_jobs[
            st.session_state.df_personalize_jobs["jobName"] == selected_job_name
        ]

        # Get the batchSegmentJobArn corresponding to the selected job name
        selected_job_arn = selected_job["batchSegmentJobArn"].values[0]

        # Fetch the personalize job details using the selected job ARN
        personalize_batch_segment_job = get_personalize_job(selected_job_arn)

        # Parse the returned JSON
        job_details = json.loads(personalize_batch_segment_job.decode("utf-8"))

        # Extract the S3 path for the jobOutput
        s3_path = job_details["batchSegmentJob"]["jobOutput"]["s3DataDestination"][
            "path"
        ]

        # Get the job name
        job_name = job_details["batchSegmentJob"]["jobName"]

        # Add the S3 file name to the path
        s3_file_path = f"{s3_path}{job_name}.json.out"

        try:
            # Read the file content
            file_content = read_s3_file(s3_file_path)

            # Process the content to get the DataFrame
            df_recommended_segments = process_json_content(file_content)
        except Exception:
            print(Exception)
            st.error("No Segment Export File Found. Is your Segment Export Job ACTIVE?")

        # TODO
        # For now just take demo data
        with fs.open(f"s3://{BUCKET_NAME}/demo-data/df_segment_data.csv", "rb") as f:
            user_data = pd.read_csv(f)
        # Convert both columns to the same data type (e.g., string)
        df_recommended_segments["userId"] = df_recommended_segments["userId"].astype(
            str
        )
        user_data["User.UserId"] = user_data["User.UserId"].astype(str)
        combined_df = df_recommended_segments.merge(
            user_data, left_on="userId", right_on="User.UserId", how="left"
        )
        st.write("### Recommended Customers Information")
        st.write(combined_df)
        st.button(
            "Confirm to use this Segment Data",
            on_click=save_df_session_state(
                df=combined_df, df_name=f"(Personalize)-{job_name}"
            ),
        )

    #########################
    #       GET ALL PERSONALIZE BATCH SEGMENT JOBS AND SHOW IN TABLE
    #########################

    st.divider()

    st.markdown("### Segment Jobs Status")

    # Create a placeholder for the dataframe on the main page
    jobs_df_main_placeholder = st.empty()

    # Add a button to fetch all segment jobs
    if st.button("Fetch Segment Jobs"):
        # Fetch the personalize jobs when the button is pressed
        personalize_jobs = cached_get_personalize_jobs()
        # Decode the byte string and parse as JSON
        data = json.loads(personalize_jobs.decode("utf-8"))
        st.session_state.df_personalize_jobs = pd.DataFrame(data["batchSegmentJobs"])
        st.session_state.df_personalize_jobs = (
            st.session_state.df_personalize_jobs.sort_values(
                by="creationDateTime", ascending=False
            )
        )
        # Update the placeholder on the main page with the new dataframe
        jobs_df_main_placeholder.dataframe(
            st.session_state.df_personalize_jobs[["jobName", "status"]],
            hide_index=True,
            use_container_width=True,
        )

import streamlit as st
import pandas as pd
import langchain as lc
from langchain import PromptTemplate
from langchain.llms.bedrock import Bedrock
import datetime
import os
import sys
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
import logging
from streamlit_extras.switch_page_button import switch_page
import s3fs
from components.utils_models import BEDROCK_MODELS

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
TITLE = "Prompt Iterator"
DESCRIPTION = "Iterate on your Prompt using Prompt Engineering and Auto-Prompting"
PAGE_TITLE = "Prompt Iterator"
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

# answer to display then there are no references
DEFAULT_NEGATIVE_ANSWER = "Could not answer based on the provided documents. Please rephrase your question, reduce the relevance threshold, or ask another question."  # noqa: E501

# default hello message
HELLO_MESSAGE = "Hi! I am an AI assistant. How can I help you?"

# page name for caching
PAGE_NAME = "ai_chat"

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

# Initialize session state
if "prompt" not in st.session_state:
    st.session_state.prompt = ""

########################################################################################################################################################################
######################################################## Session States and CSS      ###################################################################################
########################################################################################################################################################################


# define what option labels and icons to display
option_data = [
    {"icon": "bi bi-hand-thumbs-up", "label": "Agree", "color": "green"},
    {"icon": "fa fa-question-circle", "label": "Unsure"},
    {"icon": "bi bi-hand-thumbs-down", "label": "Disagree"},
]

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

########################################################################################################################################################################
######################################################## PAGE CODE    ##################################################################################################
########################################################################################################################################################################


#########################
#       SIDEBAR MODEL SELECTION
#########################
with st.sidebar:
    st.markdown("")

    # language model
    st.subheader("Language Model")
    ai_model = st.selectbox(
        label="Select a language model:",
        options=MODELS_DISPLAYED,
        key="ai_model",
        # on_change=run_genai_query,
        help="Choose the LLM model used for content generation",
    )
    if st.session_state["ai_model"] in MODELS_UNAVAILABLE:
        st.error(f'{st.session_state["ai_model"]} not available', icon="⚠️")
        st.stop()
    elif st.session_state["ai_model"] in MODELS_NOT_DEPLOYED:
        st.error(f'{st.session_state["ai_model"]} has been shut down', icon="⚠️")
        st.stop()


#########################
#       PAGE CONTENT
#########################
# Link to auto-prompting blog
st.markdown(
    """
    *For more details on prompt engineering and auto-prompting, visit [our blog](https://medium.com/@philippkai/from-prompt-engineering-to-auto-prompt-optimisation-d2de596d87e1). 
    """
)

# Create a large text box for prompt input
user_input = st.text_area(
    "Enter your prompt here:", value=st.session_state.prompt, height=300
)

# Confirm prompt button
if st.button("Confirm Prompt"):
    st.session_state.prompt = user_input
    st.success(f"Prompt saved: {st.session_state.prompt}")

# Display saved prompt
if st.session_state.prompt:
    st.sidebar.write(f"Saved Prompt: {st.session_state.prompt}")


# Sample Prompt for Airlines
st.write("## Sample Airlines Prompt")
st.write(
    """My name is John Smith. I will promote flight ticket of Airline, from SRCCity to DSTCity, during Season 2023 for membership. The ticket original price is DynamicPrice, discount is DiscountForMember for member only, for example, a discount of 0.5 means 50%, promotion code is ITEM_ID, only show the last 5 digits of ITEM_ID. Booking website is https://demobooking.demo.co, please also help to suggest itinerary details for DurationDays days in the destination city, with an attractive title to help me to promote the flight ticket to the customers. Write a high-converting {channel} to captivate the customer {name} - aged {age} years old.
         """
)

st.write("## Sample Banking Prompt")
st.write(
    """<INST>You are a marketing content creator assistant for AnyCompany Bank, a respectable financial institution with a reputation for being trustworthy and factual. You are assisting John Smith, a 54 year old bank advisor who has worked at AnyCompany Bank for 20 years. John's contact number is +1 (555) 555-1234 and his email is john@anycompany.com.

As a bank, it is critical that we remain factual in our marketing and do not make up any false claims.

Please write a {channel} marketing message to promote our new product to {name}.

The goal is to highlight the key features and benefits of the product in a way that resonates with customers like {name}, who is {age} years old.

When writing this {channel} message, please adhere to the following guidelines:

Only use factual information provided in the product description. Do not make up any additional features or benefits.
Emphasize how the product can help meet the typical financial needs of customers like {name}, based on their age demographic.
Use a warm, helpful tone that builds trust and demonstrates how AnyCompany Bank can assist customers like {name}.
If unable to compose a factual {channel} message for {name}, indicate that more information is needed. As a respectable bank, AnyCompany Bank's reputation relies on marketing content that is honest, accurate and trustworthy. Please ensure the {channel} message aligns with our brand identity. Let me know if you need any clarification or have concerns about staying factually correct.
</INST>
         """
)

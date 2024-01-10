from dotenv import load_dotenv

load_dotenv()

import json
import os

from langchain.agents import create_pandas_dataframe_agent, AgentType
from langchain.chains import APIChain
from langchain.chat_models import ChatOpenAI
import pandas as pd
import requests
import streamlit as st

from utils.trips import CompanyTrip


COMPANIES = ("shadazzle", "fitlyfe", "socimind")

# If you return an object on a line by itself in Streamlit,
# it will render to the screen in markdown
"# Playground"

# This is using a streamlit widget [selectbox](https://docs.streamlit.io/library/api-reference/widgets/st.selectbox)
# Notice that I am storing the result in a variable that will always be updated on change
company = st.selectbox(
    "Choose an existing Company Trip",
    options=COMPANIES,
    format_func=lambda x: x.title(),
    key="company",
)


# If you want to cache things so that it survives a re-render, 
# this is a good way to memoize
@st.cache_resource
def get_company_trip(company_name):
    return CompanyTrip.from_name(company_name)

# I've abstracted out the LLM conversation that created the company
# You can use this object to ask a question to the model it's in [utils/trips](./utils/trips.py)
company_trip = get_company_trip(company)

# This is where the fake APIs are hosted
# You can append ?_limit=10&_page=2 https://github.com/typicode/json-server#paginate
profiles_url = f"https://llm-companies.cyclic.app/api/{company}/profiles"

# Again this variable here will be updated when the [slider](https://docs.streamlit.io/library/api-reference/widgets/st.slider) changes.
browse_id = st.slider(f"Browse profiles for {company.title()}", max_value=50)

if browse_id > 0:
    response = requests.get(f"{profiles_url}/{browse_id}")
    profile = response.json()
    # `st.code` generates a nice color coded block!
    st.code(json.dumps(profile, indent=4))


"## Using the `CompanyTrip` helper to load previous chat history"

# This is a context manager for forms and why it's using this `with` pattern
with st.form("company-prompt"):
    company_prompt = st.text_area(f"This prompt is in context for {company}")
    # Submitted will be true if they pressed the button!
    # This is powerful because you can only re-render when they press the button
    submitted = st.form_submit_button("Ask")
    if submitted:
        answer = company_trip.ask(company_prompt)
        st.markdown(answer)

"""## Using the Profiles API documentation

This is using [LangChain's APIChain](https://python.langchain.com/en/latest/modules/chains/examples/api.html) to access the documentation for the API.
"""

# I am caching this with a decorator because I only want this to run when it needs to.
# In this case on company change it will be "reactive"
@st.cache_data
def get_company_metadata(company):
    response = requests.get(f"https://llm-companies.cyclic.app/api/{company}/metadata")
    return response.json()


metadata = get_company_metadata(company)

# `st.expander` is such a nice widget experience!
# Use it for spoilers!
with st.expander("API Docs"):
    st.code(metadata["docs"])


with st.form("profiles-api"):
    profiles_prompt = st.text_area(f"What would you like to ask the profiles API?")
    submitted = st.form_submit_button("Use API")
    if submitted:
        docs = metadata["docs"]
        llm = ChatOpenAI(model_name=os.environ["OPENAI_MODEL"], temperature=0)
        # This is in the notes above that are rendered to the screen
        api_chain = APIChain.from_llm_and_api_docs(llm, docs, verbose=True)
        result = api_chain.run(profiles_prompt)
        st.write(result)

# This is just Craig hacking around. Feel free to let it inspire you or ignore it!
# Also note how nice this conditional display is!
if company == "shadazzle":

    """## Experimenting with other data
    
This is using LangChain's [pandas DataFrame Agent](https://python.langchain.com/en/latest/modules/agents/toolkits/examples/pandas.html)
    
"""
    # I used `CompanyTrip` to build some fake reviews and stored them in json in ./trips/data/shadazzle-reviews.json
    # Read the fake reviews json to create a Pandas data frame
    # If pandas is new to you, I'm sure someone would love to help!
    reviews_df = pd.read_json(os.path.join("trips", "data", "shadazzle-reviews.json"))

    f"## First 5 of {len(reviews_df)} reviews"
    # If you put a value on a single line, like a Pandas dataframe it will just render it
    reviews_df[:5]

    # Caching the agent, not sure this is needed
    @st.cache_resource
    def get_reviews_agent():
        # Create a LangChain agent that can query the dataframe using 
        # OpenAI
        return create_pandas_dataframe_agent(
            ChatOpenAI(model_name=os.environ["OPENAI_MODEL"], temperature=0),
            reviews_df,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
        )

    reviews_agent = get_reviews_agent()

    with st.form("query_reviews"):
        reviews_query = st.text_area(
            "What would you like to know about this Reviews dataset?"
        )
        submitted = st.form_submit_button("Ask")
        if submitted:
            response = reviews_agent.run(reviews_query)
            print(f"Response from agent: {response}")
            st.write(response)

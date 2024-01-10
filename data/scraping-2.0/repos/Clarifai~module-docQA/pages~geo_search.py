import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from clarifai.client.auth.helper import ClarifaiAuthHelper
from clarifai.client import create_stub
from clarifai.modules.css import ClarifaiStreamlitCSS
from langchain import LLMChain, PromptTemplate
from langchain.llms import Clarifai

from utils.geo_search_utils import (display_location_info, get_location_data,
                                    get_summarization_output, llm_output_to_json,
                                    process_post_searches_response, search_with_geopoints)
from utils.prompts import NER_LOC_RADIUS_PROMPT

# Set Streamlit page configuration
st.set_page_config(
    page_title="GEOINT NER Investigation",
    page_icon="https://clarifai.com/favicon.svg",
    layout="wide",
)

ClarifaiStreamlitCSS.insert_default_css(st)

USER_ID = "openai"
APP_ID = "chat-completion"
MODEL_ID = "GPT-3_5-turbo"

# Authenticate user and get stub for Clarifai API
auth = ClarifaiAuthHelper.from_streamlit(st)
stub = create_stub(auth)
userDataObject = auth.get_user_app_id_proto()

# Get user's task query
task_query = st.text_area("Enter your task here")

# If task_query is not empty
if task_query:
  # Create OpenAI language model
  pat = auth._pat
  llm_chatgpt = Clarifai(pat=pat, user_id=USER_ID, app_id=APP_ID, model_id=MODEL_ID)
  # Create prompt template that retrieves the location and radius from the task query
  prompt = PromptTemplate(template=NER_LOC_RADIUS_PROMPT, input_variables=["page_content"])
  llm_chain = LLMChain(prompt=prompt, llm=llm_chatgpt)
  # Run language model chain to get location object from task query
  chain_output = llm_chain(task_query)
  chain_output_json = llm_output_to_json(chain_output["text"])
  location_obj = get_location_data(chain_output_json["LOC"])

  # If location object is found, display address, latitude, longitude, and radius
  if location_obj is not None:
    display_location_info(location_obj, chain_output_json["RADIUS"])
  # If location object is not found, display error message
  else:
    st.error(f"Coordinates not found for this location: {chain_output_json['LOC']}")
    st.stop()

  # Search posts with geopoints using Clarifai API
  post_searches_response = search_with_geopoints(
      stub,
      userDataObject,
      location_obj.longitude,
      location_obj.latitude,
      float(chain_output_json["RADIUS"]),
  )

  # Process post search response into a dictionary list
  input_dict_list = process_post_searches_response(auth, post_searches_response)
  # Convert dictionary list to pandas DataFrame
  input_df = pd.DataFrame(input_dict_list)

  # If DataFrame is empty, display warning message
  if input_df.empty:
    st.warning("No searches found for this query")

  # If DataFrame is not empty, proceed with displaying and summarizing searches
  else:
    # generate two sets of random noise with different means and standard deviations
    random_noise_1 = np.random.normal(loc=0.0, scale=0.1, size=len(input_df))
    random_noise_2 = np.random.normal(loc=0.0, scale=0.2, size=len(input_df))

    # add random column to latitude and longitude and remove random column
    input_df["lat"] = input_df["lat"] + np.where(random_noise_1 > 0, random_noise_1,
                                                 random_noise_2)
    input_df["lon"] = input_df["lon"] + np.where(random_noise_1 > 0, random_noise_2,
                                                 random_noise_1)

    st.dataframe(input_df)

    # Create a map scatter map plot of the search results
    fig = px.scatter_mapbox(
        input_df,
        lat="lat",
        lon="lon",
        zoom=3,
        hover_name="source",
        hover_data=["input_id", "page_number", "page_chunk_number"],
        color_discrete_sequence=["red"],
        height=800,
        width=800,
        template="plotly_white",
    )

    # Update the map style and layout
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    # Display the map plot in the Streamlit app
    st.plotly_chart(fig, use_container_width=True)

    if st.button("Summarize Searches"):
      texts = input_df["text"].to_list()
      text_summary = get_summarization_output(texts)
      st.write(text_summary)

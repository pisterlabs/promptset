import json
import traceback
from typing import Dict, List, Optional

import geopy
import requests
import streamlit as st
from clarifai.client.auth.helper import ClarifaiAuthHelper
## Import in the Clarifai gRPC based objects needed
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from geopy.geocoders import Nominatim
from grpc import RpcError
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.llms import Clarifai

geolocator = Nominatim(user_agent="geoint")

USER_ID = "openai"
APP_ID = "chat-completion"
MODEL_ID = "GPT-3_5-turbo"

# Create function that searches with a given longitude and latitude
@st.cache_resource
def search_with_geopoints(
    _stub,
    userDataObject: resources_pb2.UserAppIDSet,
    longitude: float,
    latitude: float,
    radius: float,
) -> Optional[service_pb2.MultiSearchResponse]:
  try:
    post_searches_response = _stub.PostSearches(
        service_pb2.PostSearchesRequest(
            user_app_id=userDataObject,
            query=resources_pb2.Query(ands=[
                resources_pb2.And(input=resources_pb2.Input(data=resources_pb2.Data(
                    geo=resources_pb2.Geo(
                        geo_point=resources_pb2.GeoPoint(
                            longitude=longitude,
                            latitude=latitude,
                        ),
                        geo_limit=resources_pb2.GeoLimit(type="withinKilometers", value=radius),
                    ))))
            ]),
        ),)

    if post_searches_response.status.code != status_code_pb2.SUCCESS:
      print(post_searches_response)
      st.error("Post searches failed, status: " + post_searches_response.status.description)

    print("Found inputs:")
    print(len(post_searches_response.hits))
    return post_searches_response

  except RpcError as e:
    st.error(f"Error: {_stub} connection error.")
    st.stop()  # stop the streamlit app if there's a connection error.
  except Exception as e:
    st.error(f"Error: {e} - {traceback.print_exc()}")



def url_to_text(auth, url):
  try:
    h = {"Authorization": f"Key {auth.pat}"}
    response = requests.get(url, headers=h)
    response.encoding = response.apparent_encoding
  except Exception as e:
    print(f"Error: {e}")
    response = None
  return response.text if response else ""


@st.cache_resource
def process_post_searches_response(
    _auth,
    post_searches_response: service_pb2.MultiSearchResponse,
) -> List[Dict]:
  """
    Given a response object from a POST /searches API call, returns a list of
    dictionaries containing the input data for each search result that was
    successfully downloaded.
    """
  input_success_status = {
      status_code_pb2.INPUT_DOWNLOAD_SUCCESS,
      status_code_pb2.INPUT_DOWNLOAD_PENDING,
      status_code_pb2.INPUT_DOWNLOAD_IN_PROGRESS,
  }

  input_dict_list = []
  for idx, hit in enumerate(post_searches_response.hits):
    input = hit.input
    if input.status.code not in input_success_status:
      continue

    # Initializations
    input_dict = {}
    input_dict["input_id"] = input.id
    input_dict["text"] = url_to_text(_auth, input.data.text.url)
    input_dict["source"] = input.data.metadata["source"]
    input_dict["text_length"] = input.data.metadata["text_length"]
    input_dict["page_number"] = input.data.metadata["page_number"]
    input_dict["page_chunk_number"] = input.data.metadata["page_chunk_number"]
    input_dict["lat"] = input.data.geo.geo_point.latitude
    input_dict["lon"] = input.data.geo.geo_point.longitude
    input_dict_list.append(input_dict)

  return input_dict_list


@st.cache_resource
def llm_output_to_json(llm_output: str) -> Dict:
  """
    Given a string containing LLM output, returns a dictionary with the entity
    information extracted from the output.
    """
  if isinstance(llm_output, dict) and len(llm_output) == 2:
    return llm_output
  elif isinstance(llm_output, str):
    entity_dict = {}
    llm_output = llm_output.strip()
    if "output" in llm_output[:20].lower():
      llm_output = llm_output.split("Output:")[1].strip()
    if "json" in llm_output:
      llm_output = llm_output.split("json")[1].strip().split('`',1)[0]
    try:
      entity_dict = json.loads(llm_output)
    except Exception as e:
      st.error(f"output: {llm_output}")
      st.error(f"error: {e}")
    return entity_dict


def get_location_data(location_str: str) -> Optional[geopy.Location]:
  """
    Returns the geolocation object for the given location string.
    """
  try:
    location_obj = geolocator.geocode(location_str, language="en", timeout=None)
    return location_obj
  except Exception as e:
    st.error(f"Error: {e}")
    return None


def display_location_info(location_obj: geopy.Location, radius: float) -> None:
  """
    Displays information about the location in the Streamlit app.
    """
  col1, col2, col3 = st.columns(3)
  with col1:
    st.info(f"{location_obj.address}")
  with col2:
    st.info(f"LAT: {location_obj.latitude:.4f} - LON: {location_obj.longitude:.4f}")
  with col3:
    st.info(f"{radius:.2f} KM Radius")


@st.cache_resource
def get_summarization_output(texts: List[str]) -> str:
  """
    Returns the summarization output using the LLM chain.
    """
  docs = [Document(page_content=t) for t in texts]

  auth = ClarifaiAuthHelper.from_streamlit(st)
  pat = auth._pat
  llm = Clarifai(pat=pat, user_id=USER_ID, app_id=APP_ID, model_id=MODEL_ID)
  summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
  text_summary = summary_chain.run(docs)
  return text_summary

import streamlit as st
import os
# import logging
# import sys
# import time
# import boto3
# from botocore.exceptions import ClientError
# import requests


import assemblyai as aai
import requests
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate


aai.settings.api_key = f"c3e657dcbb774a30b06d4b300e28535c"
transcriber = aai.Transcriber()

prompts = {}
inputs = {} #used to merge into prompt templates, merged into the "{user_input}" placeholder
defaults = {} #used for default values in simple examples


prompts[""] = """
{user_input}

According to the transcription above, suggest recommendation on what the customer support agent can do beter.
"""

def get_llm():
    
    model_kwargs = { #AI21
        "maxTokens": 1024, 
        "temperature": 0, 
        "topP": 0.5, 
        "stopSequences": [], 
        "countPenalty": {"scale": 0 }, 
        "presencePenalty": {"scale": 0 }, 
        "frequencyPenalty": {"scale": 0 } 
    }
    
    llm = Bedrock(
        credentials_profile_name=os.environ.get("BWB_PROFILE_NAME"), #sets the profile name to use for AWS credentials (if not the default)
        region_name=os.environ.get("BWB_REGION_NAME"), #sets the region name (if not the default)
        endpoint_url=os.environ.get("BWB_ENDPOINT_URL"), #sets the endpoint URL (if necessary)
        model_id="ai21.j2-ultra-v1", #set the foundation model
        model_kwargs=model_kwargs) #configure the properties for Claude
    
    return llm


def get_prompt(user_input, template):
    
    prompt_template = PromptTemplate.from_template(template) #this will automatically identify the input variables for the template

    prompt = prompt_template.format(user_input=user_input)
    
    return prompt


def get_text_response(user_input, template): #text-to-text client function
    llm = get_llm()
    
    prompt = get_prompt(user_input, template)
    
    return llm.predict(prompt) #return a response to the prompt



def aai_transcriber(audio_path, type):
   with st.spinner("Working..."):
      transcribe = transcriber.transcribe(audio_path).text
      summarize =transcriber.transcribe(
      audio_path, 
      config = aai.TranscriptionConfig(
         summarization=True, 
         summary_model=aai.SummarizationModel.informative,
         summary_type=aai.SummarizationType.bullets)).summary

      improvement = get_text_response(user_input=transcribe, template=prompts[""])
      return f"Transcription:\n{transcribe}\n\nSummarization\n{summarize}\n\nRecommendation\n{improvement}"


def load_view():
    type=""
    audio_path="none"
    st.title("Transcription Improvement")

    # Create two columns; adjust the ratio to your liking
    col1, col2 = st.columns([4,1]) 

    # Use the first column for text input
    with col1:
        input_text = st.text_input(label="file link", key="audio_textfield", value="https://github.com/diminecjean/weienlooi/raw/main/SampleCall.mp3", label_visibility="collapsed")
    # Use the second column for the submit button
    with col2:
        transcribe_button = st.button("Run", type="primary", key="transcribe_button")

    
    uploaded_file = st.file_uploader("Choose an audio file", type=["mp3"])

    if uploaded_file is not None:
        audio_path=uploaded_file
    elif transcribe_button:
        audio_path=input_text


    if (audio_path!="none"):
        st.write(aai_transcriber(audio_path,type))
    else:
        st.write("")


# # Add relative path to include demo_tools in this code example without need for setup.
# sys.path.append("../..")
# # from demo_tools.custom_waiter import CustomWaiter, WaitState

# logger = logging.getLogger(__name__)

# def start_job(
#     job_name,
#     media_uri,
#     media_format,
#     language_code,
#     transcribe_client,
#     vocabulary_name=None,
# ):
#     """
#     Starts a transcription job. This function returns as soon as the job is started.
#     To get the current status of the job, call get_transcription_job. The job is
#     successfully completed when the job status is 'COMPLETED'.

#     :param job_name: The name of the transcription job. This must be unique for
#                      your AWS account.
#     :param media_uri: The URI where the audio file is stored. This is typically
#                       in an Amazon S3 bucket.
#     :param media_format: The format of the audio file. For example, mp3 or wav.
#     :param language_code: The language code of the audio file.
#                           For example, en-US or ja-JP
#     :param transcribe_client: The Boto3 Transcribe client.
#     :param vocabulary_name: The name of a custom vocabulary to use when transcribing
#                             the audio file.
#     :return: Data about the job.
#     """
#     try:
#         job_args = {
#             "TranscriptionJobName": job_name,
#             "Media": {"MediaFileUri": media_uri},
#             "MediaFormat": media_format,
#             "LanguageCode": language_code,
#         }
#         if vocabulary_name is not None:
#             job_args["Settings"] = {"VocabularyName": vocabulary_name}
#         response = transcribe_client.start_transcription_job(**job_args)
#         job = response["TranscriptionJob"]
#         logger.info("Started transcription job %s.", job_name)
#     except ClientError:
#         logger.exception("Couldn't start transcription job %s.", job_name)
#         raise
#     else:
#         return job

# def load_view():
#     logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

#     transcribe_client = boto3.client("transcribe")
#     st.title("Transcription Improvement")

#     # Create two columns; adjust the ratio to your liking
#     col1, col2 = st.columns([4,1,]) 

#     # Use the first column for text input
#     with col1:
#         input_text = st.text_input(label="file link", key="audio_textfield", value="https://github.com/AssemblyAI-Examples/audio-examples/raw/main/20230607_me_canadian_wildfires.mp3", label_visibility="collapsed")
#     # Use the second column for the submit button
#     with col2:
#         transcribe_button = st.button("Transcribe", type="primary", key="transcribe_button")
    
#     uploaded_file = st.file_uploader("Choose an audio file", type=["mp3"])

#     if transcribe_button:
#         with st.spinner("Working..."):
#             st.write(start_job("test", input_text, "mp3", "en-US", transcribe_client))

      


         
      
      
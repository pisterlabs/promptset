# #################################################################
# This runs the entire transcription process
# #################################################################



# # https://github.com/linto-ai/whisper-timestamped

import whisper_timestamped as whisper
import json

# from langchain.document_loaders import PyPDFLoader
# from langchain.llms import OpenAI
# from langchain.prompts import PromptTemplate
# from langchain.schema import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chat_models import ChatOpenAI
# from langchain.vectorstores import FAISS
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.chains.summarize import load_summarize_chain
# import numpy as np
# from sklearn.cluster import KMeans
# from langchain.document_loaders import TextLoader, UnstructuredAPIFileIOLoader
import json
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# from langchain.output_parsers import StructuredOutputParser, ResponseSchema
# from langchain.embeddings import HuggingFaceEmbeddings
# from local_functions import contagion_simulation, check_errors
import pickle
import time
import os

os.environ["OPENAI_API_KEY"] = "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"


# Record the start time
start_time = time.time()

# eric_sprott
# all_in_rkj_interview
# david_lin_interview_doomberg
# david_lin_interview_rosenburg
# forward_guidance_micheal_pettis
# interview_santiago_xrp
# jay_martin_interview_luke_gromen
# kazatomprom_interview
# rick_rule

file_name = 'admiral_mcraven_speech.mp3'
# Set up the input and output folders
audio_mp3_input_folder = "audio_mp3_input"
audio_transcript_input_folder = "audio_transcript_input"
title_timestamped_folder = "audio_title_timestamped_input"



# #################################################################
# Creating the audio transcript
# #################################################################

# Load the model
model = whisper.load_model("tiny", device="cpu")



if not os.path.exists(audio_transcript_input_folder):
    os.mkdir(audio_transcript_input_folder)


audio_file = os.path.join(audio_mp3_input_folder, file_name)
audio = whisper.load_audio(audio_file)

# Transcribe the audio and split the text into segments
result = whisper.transcribe(model, audio, language="en")


# #################################################################
# Extracting only the neccessary audio transcript and discarding the rest
# #################################################################

segments = result['segments']


output = {
    "all_text": ' '.join([item['text'] for item in segments]),
    "segments": []
}

print (output['all_text'])


# Extract the segment data
for segment in segments:
    id = segment['id']
    start = segment['start']
    end = segment['end']
    text = segment['text']
    output["segments"].append({
        "id": id,
        "start": start,
        "end": end,
        "text": text
    })




# # If there are multiple repeated words or characters - exit the program.
# repeated_words = check_errors(output['all_text'])

# #################################################################
# Output the audio transcript to folder
# #################################################################

output_file_name = "output_timestamp_" + os.path.splitext(file_name)[0] + ".json"
output_file = os.path.join(audio_transcript_input_folder, output_file_name)

# Write the final output to a JSON file
with open(output_file, 'w') as f:
    json.dump(output, f)

print(f"Transcription complete for {audio_file}. Output file saved to {output_file}.")



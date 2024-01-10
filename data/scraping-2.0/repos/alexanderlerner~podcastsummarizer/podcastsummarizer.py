import yaml
import pandas as pd
import re

import feedparser
import urllib
import argparse

import whisper
import os
import openai
from fpdf import FPDF
import base64
# For writing to PDF
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph

# Streamlit
import streamlit as st

st.write("""
# Podcast Summarizer
Each podcast summary may take a few minutes.
""")

# Load the config file which contains user inputs
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
f.close()

# Load the podcasts.txt file which cointains the RSS Links of podcasts to summarize
with open('podcasts.txt', 'r') as f:
    links = f.readlines()
f.close()

# Get Open AI API Key, Specific Chat Model for Summarization, and Whisper Model
OPEN_AI_KEY = config['open_ai_key']
OPEN_AI_MODEL = config['gpt_model']
WHISPER_TYPE = config['whisper_model']

# Get predefined chunk size depending on model constraints
CHUNK_SIZE = config['chunk_size']

# Get podcast RSS links
RSS_LINKS = links

SUMMARY_PROMPT = config['summary_prompt']
CHUNK_PROMPT = config['chunk_prompt']

# Get Podcast MP3 Files from RSS Links (From Andrew)
def download_rss_feed(rss_url: str ,numpcast: int):
    # download the rss feed
    feed = feedparser.parse(rss_url)
    # get the url of the mp3 file
    entry = feed.entries[numpcast]
    title = entry.title
    mp3_url = entry.enclosures[0].href
    title = re.sub(r'[\\/:*?"<>|]', '_', title)
    title = "".join(title.strip())
    title = re.sub(r'[^a-zA-Z0-9\s]', '', title).replace('_', ' ')
    save_path = title + ".mp3"
    # download the mp3 file
    print(f"save_path: {save_path}")
    urllib.request.urlretrieve(mp3_url,save_path)
    return title

# Use Whisper to generate a transcript of the podcast
def generate_podcast_transcript(rss_link, whisper_model):
  title = download_rss_feed(rss_link,0)
  save_path = title + ".mp3"
  result = whisper_model.transcribe(save_path, verbose = False)
  return (title, result['text'])

def split_transcript(text, max_wordcount):
  """
  Here we split the podcast transcript into a list of chunks, so that
  if a podcast transcript is greater than a desired maximum wordcount(param)
  then we split it into chunks of size max_wordcount. We'll use this to handle
  max token constraints for different models we try out.
  ------------------------------------
  Returns each chunk of the podcast transcript in a list
  """
  words = text.split()
  length = len(words)
  text_chunks = []
  for i in range(0, length, max_wordcount):
      subset_words = words[i:i+max_wordcount]
      transcript = ' '.join(subset_words)
      text_chunks.append(transcript)
  return text_chunks  

def generate_chunk_summaries(SUMMARY_PROMPT, OPEN_AI_MODEL, chunks):
    """
    Query an OpenAI model to create a summary of each chunk of the podcast transcript.
    Uses the OpenAI Chat Completion API. More details on the API here: https://platform.openai.com/docs/guides/gpt
    ------------------------------------
    Returns the summaries of each chunk in a list. 
    """
    summaries = []
    for i in range(len(chunks)):
        response = openai.ChatCompletion.create(
        model= OPEN_AI_MODEL,
            messages=[
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": chunks[i]}
        ])
        summaries.append(response['choices'][0]['message']['content']) 
    return summaries

def generate_final_summary(CHUNK_PROMPT: str, OPEN_AI_MODEL: str, compiled_summaries: str) -> str:
    """
    Query an OpenAI model to create the final summary using the previous summaries of each chunk.
    Relies on user-inputted CHUNK_PROMPT to query the model, and the desired OPEN_AI_MODEL. 
    Please see config.yaml to modify.
    More details on the Chat Completion API here: https://platform.openai.com/docs/guides/gpt
    ------------------------------------
    Returns the final summary of the podcast.
    """
    response = openai.ChatCompletion.create(
    model=OPEN_AI_MODEL,
    messages=[
      {"role": "system", "content": CHUNK_PROMPT},
      {"role": "user", "content": compiled_summaries}
        ])

    return response['choices'][0]['message']['content']

"""
Generate the Summary for each Podcast.
"""

# Set up Whisper Model
whisper_model = whisper.load_model(WHISPER_TYPE)
# Set Up OpenAI Access
openai.api_key = OPEN_AI_KEY

summaries = {}
n_podcasts = len(RSS_LINKS)

for i, link in enumerate(RSS_LINKS):
    
    title, transcript = generate_podcast_transcript(link, whisper_model)
    
    # Remove mp3 file after we get the transcript
    os.remove(title+".mp3")
    # Split full transcript into a sequential list of 1000 word chunks
    chunks = split_transcript(transcript, CHUNK_SIZE)
    # Generate a Summary for each chunk using the user-inputed OPEN AI model
    chunk_summaries = generate_chunk_summaries(SUMMARY_PROMPT, OPEN_AI_MODEL, chunks)
    # Get all the Chunk Summaries into one string
    compiled_summaries = ' '.join(chunk_summaries)
    # Create the final summary using all the previous chunks
    final_summary = generate_final_summary(CHUNK_PROMPT, OPEN_AI_MODEL, compiled_summaries)
    st.success("")
    st.write("## " + title)
    st.write(final_summary)
    print(f"Generated Summary {i+1}/{n_podcasts}")
    summaries[title] = final_summary

"""
Export/Write to PDF
2 Options: Write to a PDF Directly, or take the streamlit and export to PDF.
"""

# Write to PDF

def write_dict_to_pdf(data_dict, output_path):
    # Create a PDF document
    doc = SimpleDocTemplate(output_path, pagesize=letter)

    # Create paragraph styles for the title and summary
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    summary_style = ParagraphStyle('Summary', parent=styles['Normal'], spaceAfter=10)

    # Create a list to hold the elements of the PDF
    elements = []

    # Iterate over the dictionary items
    for title, summary in data_dict.items():
        # Create a paragraph for the title with bold font
        title_paragraph = Paragraph(f'<b>{title}</b>', title_style)
        elements.append(title_paragraph)

        # Create a paragraph for the summary
        summary_paragraph = Paragraph(summary, summary_style)
        elements.append(summary_paragraph)

    # Build the PDF document
    doc.build(elements)
# export_as_pdf = st.button("Export to PDF")

# def create_download_link(val, filename):
#     b64 = base64.b64encode(val)  # val looks like b'...'
#     return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'

# if export_as_pdf:
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font('Arial', 'B', 16)
#     pdf.cell(40, 10, report_text)
    
#     html = create_download_link(pdf.output(dest="S").encode("latin-1"), "test")
#     st.markdown(html, unsafe_allow_html=True)
output_file = "podcast_summary.pdf"
write_dict_to_pdf(summaries, output_file)



    


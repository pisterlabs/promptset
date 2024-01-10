# Databricks notebook source
"""Document Loading
Retrieval augmented generation
In retrieval augmented generation (RAG), an LLM retrieves contextual documents from an external dataset as part of its execution.

This is useful if we want to ask question about specific documents (e.g., our PDFs, a set of videos, etc).



"""

# COMMAND ----------

#! pip install langchain
import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']


# COMMAND ----------

"""PDFs
Let's load a PDF transcript from Andrew Ng's famous CS229 course! These documents are the result of automated transcription so words and sentences are sometimes split unexpectedly."""
# The course will show the pip installs you would need to install packages on your own machine.
# These packages are already installed on this platform and should not be run again.
#! pip install pypdf 


# COMMAND ----------

from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")
pages = loader.load()

# COMMAND ----------

"""Each page is a Document.

A Document contains text (page_content) and metadata."""

len(pages)

# COMMAND ----------

page = pages[0]

# COMMAND ----------

print(page.page_content[0:500])

# COMMAND ----------

page.metadata

# COMMAND ----------

from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader

# COMMAND ----------

# ! pip install yt_dlp
# ! pip install pydub

# COMMAND ----------

url="https://www.youtube.com/watch?v=jGwO_UgTS7I"
save_dir="docs/youtube/"
loader = GenericLoader(
    YoutubeAudioLoader([url],save_dir),
    OpenAIWhisperParser()
)
docs = loader.load()

# COMMAND ----------

docs[0].page_content[0:500]

# COMMAND ----------

from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://github.com/basecamp/handbook/blob/master/37signals-is-you.md")

# COMMAND ----------

docs = loader.load()

# COMMAND ----------

print(docs[0].page_content[:500])

# COMMAND ----------

"""Notion
Follow steps here for an example Notion site such as this one:

Duplicate the page into your own Notion space and export as Markdown / CSV.
Unzip it and save it as a folder that contains the markdown file for the Notion page."""

# COMMAND ----------

from langchain.document_loaders import NotionDirectoryLoader
loader = NotionDirectoryLoader("docs/Notion_DB")
docs = loader.load()

# COMMAND ----------

print(docs[0].page_content[0:200])

# COMMAND ----------

docs[0].metadata

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



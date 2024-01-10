import configparser
import pickle
import os
import time
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.chains.mapreduce import MapReduceChain
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.output_parsers import PydanticOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_extraction_chain
from pydantic import BaseModel
import openai
import glob

# Load the configuration
config = configparser.ConfigParser()
config.read('modules/suite_config.ini')

# Get the OpenAI API key from the configuration file
os.environ['OPENAI_API_KEY'] = config['OPENAI']['OPENAI_API_KEY']

# Load the data from the pickle file
with open('cache/test_weekly_cache.pkl', 'rb') as f:
    raw_documents_dict = pickle.load(f)

# Combine the news texts from all days into one list
raw_documents = [(news[0], news[2]) for day_news in raw_documents_dict.values() for news in day_news]
# Define a Document class with a page_content and metadata attribute
class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata
# Combine the news texts from all days into one list and store additional info in metadata
raw_documents = []
for week_number, week_data in raw_documents_dict.items():
    for day_name, day_news in week_data.items():
        for news in day_news:
            headline = news[0]
            if isinstance(news[4], time.struct_time):  # Checking index 4
                date_time = time.strftime("%Y-%m-%d %H:%M:%S", news[4])  # Convert time.struct_time to string
            elif isinstance(news[4], str):  # Checking index 4
                date_time = news[4]  # Use it directly
            else:
                date_time = 'N/A'  # Placeholder string
            link = news[3]
            summary = news[2]
            metadata = {'headline': headline, 'date_time': date_time, 'link': link, 'summary': summary}
            raw_documents.append((news[2], metadata))

# Convert the list of strings and metadata into a list of Document objects
documents = [Document(text, metadata) for text, metadata in raw_documents]

# Split each document into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
chunks = text_splitter.split_documents(documents)

# Generate vectorized representations for each chunk
vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings())

# Define your desired data structure.
class NewsItem(BaseModel):
    headline: str
    link: str
    date_time: str
    summary: str

# Set up a parser
parser = PydanticOutputParser(pydantic_object=NewsItem)

metadata_field_info = [
    AttributeInfo(
        name="date_time",
        description="The date and time the news article was published",
        type="string",
    ),
    AttributeInfo(
        name="headline",
        description="The headline of the news article",
        type="string",
    ),
    AttributeInfo(
        name="link",
        description="The URL link to the original news article",
        type="string",
    ),
    AttributeInfo(
        name="summary",
        description="A summary of the article",
        type="string",
    ),
]

document_content_description = "Brief summary of a news article"
llm = OpenAI(temperature=0)
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    enable_limit=True,
    verbose=True,
)

# Define the extraction schema
schema = {
    "properties": {
        "topic": {"type": "string"},
    },
    "required": ["topic"],
}

# # Get a list of all files in the directory with the given prefix
files = glob.glob('super_summaries/modular_daily_script_*.txt')
# Get a list of all files in the directory with the given prefix
# files = glob.glob('weekly_scripts/weekly_script_*.txt')
# Find the most recent file
most_recent_file = max(files, key=os.path.getctime)

# Load the text from the most recent file
with open(most_recent_file, 'r') as f:
    text = f.read()

# Initialize the LLM model
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# Create a summarization prompt
summarization_prompt = PromptTemplate(
    input_variables=["text"],
    template="{text}\n\nSummarize:",
)

# Create the summarization chain
summarization_chain = load_summarize_chain(llm, chain_type="map_reduce")

extraction_chain = create_extraction_chain(schema, llm)
import re
# Define a function to clean the topics
def clean_topics(topics):
    return [{'topic': re.sub(r'\d+', '', topic['topic']).strip()} for topic in topics]

# Run the extraction chain
extracted_entities = extraction_chain.run(text)

# Clean the topics
extracted_entities = clean_topics(extracted_entities)
# Take at most 8 entities
extracted_entities = extracted_entities[:8]

# Now extracted_entities should be a list of entities
# Use each entity as a topic to run the retriever
summaries_str = ""
# Read the content from the file
with open('russo_ukranian_war_abridged.txt', 'r') as file:
    russo_ukranian_war_summary = file.read()

for entity in extracted_entities:
    relevant_documents = retriever.get_relevant_documents(entity['topic'])
    # Prepare the document info and summary for GPT
    doc_info_and_summary = ""
    for i, doc in enumerate(relevant_documents[:1]):
        doc_info_and_summary += f"\nTopic: {entity['topic']}\n"
        doc_info_and_summary += f"Headline: {doc.metadata['headline']}\n"
        doc_info_and_summary += f"Summary: {doc.metadata['summary']}\n"
        doc_info_and_summary += f"Date and Time: {doc.metadata['date_time']}\n"
        doc_info_and_summary += f"Link: {doc.metadata['link']}\n"
        doc_info_and_summary += "\n-----------------\n"

    summaries_str += f"\nTopic: {entity['topic']}\n{doc_info_and_summary}"

# # Append the content of russo_ukranian_war_abridged.txt to summaries_str
# summaries_str += f"\n{russo_ukranian_war_summary}"

print(summaries_str)

# Generate GPT response
openai.api_key = os.getenv('OPENAI_API_KEY')
response1 = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {
            "role": "system",
            "content": f"You are a cutting-edge AI assistant named 'Cortex', tasked with crafting a professional news broadcast titled, 'NewsPlanetAI', a highly trusted news program.  Your task is to read the proposed broadcast which will be supplied by the user, and elaborate (without changing the content and tone of the proposed script) on the identified topics and inject more context using the news articles provided here. If the articles are relevant to the content, you should insert the relevant information smoothly and coherently. If you insert information, PASS ONE URL AS A SOURCE, write the entire URL. \n\nRelevant News:\n{summaries_str}"
        },
        {
            "role": "user",
            "content": text  # This is the original text from the file
        }
    ],
    temperature=.5,
    max_tokens=1000,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)

response_content = response1.choices[0].message['content']
print(response_content)

# Save the response to a file
from datetime import datetime
# Replace '.' with '.\n'
response_content = response_content.replace('. ', '.\n')
date_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"super_summaries/refined/refined_weekly_super_summary_{date_string}.txt"

with open(filename, 'w', encoding='utf-8') as f:
    f.write(response_content)
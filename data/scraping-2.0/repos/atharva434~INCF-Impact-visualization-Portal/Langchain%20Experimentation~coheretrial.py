from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import cohere
from langchain.llms import Cohere
import nltk
import json

#give this model internet access for getting real time data
llm = Cohere(cohere_api_key="4aJ9yWbIrOzI2W5LZeLeIdin2AYMpkq18PffLuvi",temperature=0,max_tokens=2096)
prompt="""template="Provide 20 multiple choice questions with 4 options for each question based on the topic provided and the level of difficulty provided.\
    Also provide answer key to this questions in the form of question number: correct answer and a suitable reason for the same\
    topic = Oscar \
    level = easy","""

print(llm(prompt))
print(json.loads(llm(prompt)))
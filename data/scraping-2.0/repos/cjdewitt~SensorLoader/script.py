from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import RetrievalQA
from langchain.agents import AgentType
from langchain.agents.react.base import DocstoreExplorer
from langchain.agents import AgentType
from langchain.agents.react.base import DocstoreExplorer
from langchain.output_parsers import OutputFixingParser
from typing import Any

from langchain.output_parsers import PydanticOutputParser

from langchain.chat_models import ChatOpenAI

from pydantic import BaseModel, Field
from typing import List

from langchain import OpenAI, Wikipedia
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.agents.react.base import DocstoreExplorer

import os
import sys
import panel as pn
import asyncio
import re
from collections import Counter
from langchain.agents import initialize_agent, Tool

openai_api_key = "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"
os.environ["OPENAI_API_KEY"] = openai_api_key

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
pdf_file = "/Users/corydewitt/Desktop/Research/REU/playgrnd/PX4_devices/ICM.pdf"

# Load the documents using the PyPDFLoader
loader = PyPDFLoader(file_path=pdf_file)
pages = loader.load()

# Initialize FAISS index with OpenAIEmbeddings
faiss_index = FAISS.from_documents(pages, embeddings)

# Initialize the react agent
llm = OpenAI(temperature=0, model_name="text-davinci-002")
docstore = DocstoreExplorer(llm)

tools = [
    Tool(
        name="Search",
        func=faiss_index.similarity_search,
        description="useful for when you need to ask with search",
    ),
    Tool(
        name="Lookup",
        func=faiss_index.similarity_search,
        description="useful for when you need to ask with lookup",
    ),
]

react = initialize_agent(tools, llm, agent=AgentType.REACT_DOCSTORE, verbose=True)

def get_questions(prompt):
    response = chat([HumanMessage(content=prompt)])
    questions = [response.content.strip()]
    return questions

def get_answers(question, abstract):
    messages = [HumanMessage(content=question), HumanMessage(content=f"abstract: {abstract}")]
    response = chat(messages)
    answer = response.content.strip()
    return answer

class ParsingError(Exception):
    pass


class CustomOutputParser(PydanticOutputParser):
    parsed_output: List[dict] = Field(default_factory=list, description="Parsed output of the custom parser")
    pydantic_object: Any = Field(default=None, description="Parsed pydantic object")

    def parse(self, text):
        # Split the text into lines
        lines = text.split("\n")

        # Initialize an empty list to store the parsed output
        parsed_output = []

        # Iterate over each line
        for line in lines:
            # Split the line into words
            words = line.split()

            # Check if the line is a pin assignment
            if len(words) >= 3 and words[0].isdigit():
                # If it is, add it to the parsed output
                parsed_output.append({"pin_number": words[0], "pin_name": words[1], "pin_description": " ".join(words[2:])})

        # Create the pydantic object from the parsed output
        pydantic_object = self.construct_pydantic_object(parsed_output)

        # Return the parsed output and the pydantic object
        return {"parsed_output": parsed_output, "pydantic_object": pydantic_object}


class CustomOutputFixingParser(OutputFixingParser):
    retry_chain: Any = None  # Add this field with a default value of None

    def __init__(self, parser, llm):
        super().__init__(parser=parser, llm=llm)

    @classmethod
    def from_llm(cls, parser, llm):
        return cls(parser=parser, llm=llm)

# Construct an OutputFixingParser
new_parser = CustomOutputFixingParser.from_llm(parser=CustomOutputParser(), llm=ChatOpenAI())






def extract_toc(pdf_text):
    # Split the text into lines
    lines = pdf_text.split("\n")

    # Find the start and end of the table of contents
    start = None
    end = None
    for i, line in enumerate(lines):
        if "TABLE OF CONTENTS" in line.upper():
            start = i
        elif start is not None and line.strip() == "":
            end = i
            break

    # If we didn't find a table of contents, return an empty dictionary
    if start is None or end is None:
        return {}

    # Extract the table of contents lines
    toc_lines = lines[start:end]

    # Parse the table of contents
    toc = {}
    for line in toc_lines:
        # This regular expression looks for any number of characters
        # followed by one or more spaces, followed by one or more digits
        match = re.match(r"(.*?)\s+(\d+)$", line)
        if match:
            section = match.group(1)
            page = match.group(2)
            if section not in toc:
                toc[section] = []
            toc[section].append(page)

    return toc

def qa(loaded_documents, query, chain_type, k):
    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(loaded_documents)
    
    # Select which embeddings we want to use
    embeddings = OpenAIEmbeddings()
    
    # Create the vector store to use as the index
    db = Chroma.from_documents(texts, embeddings)
    
    # Expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    
    # Construct an OutputFixingParser
    new_parser = CustomOutputFixingParser.from_llm(parser=CustomOutputParser(), llm=ChatOpenAI())

    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True
    )

    conversation = []  # List to store the conversation history

    while True:
        user_question = input("Enter a question or type 'exit': ")
        if user_question == 'exit':
            break

        # Add the user's question to the conversation history
        conversation.append(f"Inquiry: {user_question}")

        try:
            # Perform the question-answering
            result = qa_chain({"query": user_question})

            # Get the chat bot's answer
            chat_bot_answer = result['result']
        except ParsingError:
            # Handle parsing errors
            chat_bot_answer = qa_chain.output_parser.handle_parsing_errors(result['raw_output'])

        # Add the chat bot's answer to the conversation history
        conversation.append(f"Response: {chat_bot_answer}")

        # Print and return the result
        print(f"Inquiry: {user_question}")
        print(f"Response: {chat_bot_answer}")

    return conversation

# In your main function, pass loaded_documents to the qa function
def main():
    choice = input("Enter your choice: 1. Upload your own pdf, 2. Analyze ICM-42688P datasheet, 3. Exit: ")

    if choice == '1':
        pdf_file = input("Enter the path to your pdf:")
        loader = PyPDFLoader(file_path=pdf_file)
        loaded_documents = loader.load()
        while True:
            query = input("Enter a question or type 'exit': ")
            if query == 'exit':
                break
            result = qa(loaded_documents=loaded_documents, query=query, chain_type="map_rerank", k=2)

    elif choice == '2':
        pdf_path = "/Users/corydewitt/Desktop/Research/REU/playgrnd/PX4_devices/ICM.pdf"  # NEEDS TO CHANGE PER USER
        loader = PyPDFLoader(file_path=pdf_path)
        loaded_documents = loader.load()
        while True:
            query = input("Enter a question or type 'exit': ")
            if query == 'exit':
                break
            result = qa(loaded_documents=loaded_documents, query=query, chain_type="map_rerank", k=2)

    elif choice == '3':
        sys.exit(1)

if __name__ == "__main__":
    main()

import os
import openai
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import json

class ResumeExtractor(object):
    def __init__(self, path : str) -> None:

        # Precondition checks.
        if path == "" or path == None:
            raise ValueError(f"The path provided: {path} is not a valid path.")

        self.path = path
        loader = UnstructuredPDFLoader(path)
        self.pages  = loader.load_and_split()
        if (len(self.pages) > 3):
            raise ValueError(f"The resume provided has more than 3 pages. Please send a resume with <= 3 pages.")
        embeddings = OpenAIEmbeddings()

        # Just one page resumes accepted.
        self.docsearch = Chroma.from_documents(self.pages, embeddings).as_retriever(search_kwargs={ "k": 1 })
        self.chain = load_qa_chain(OpenAI(temperature=0, max_tokens=2500), chain_type="stuff")

    def ask(self, question : str) -> str:
        docs = self.docsearch.get_relevant_documents(question)
        output = self.chain.run(input_documents=docs, question=question)
        return output

    def extract_details(self) -> str:
        query_to_extract_info = """Using the document, answer the following questions and output valid json with property names enclosed with double quotes with keys: "is_resume", "skills", "years_of_experience", "experience_summary", "achievements", "highest_education":

        1. Is this document of a resume? Answer in "True" or "False". The answer should correspond to the "is_resume" key.
        2. What are the candidates skills? The answer should be a json list associated with the "skills" key.
        3. How many years of experience does the candidate have? The answer should correspond to the "years_of_experience" key.
        4. Based on the candidate's experience, extract achievements that are backed by numbers that the candidate has made in the form of a json list associated with the "achievements" key.
        5. What is the candidate's highest education and field of study?"""
        return self.ask(query_to_extract_info)

class CoverLetterGenerator(object):
    def __init__(self, resume_details : json):
        self.resume_details = resume_details
        self.messages = [{"role": "system", "content" : "You are a sophisticated career advisor who is helping individuals write cover letters on the basis of their resumes."}]

    def get_coverletter(self) -> dict:

        # Comparison 1.
        query_to_extract_info = f"""Based on just the json object of resume details: {self.resume_details},
        Generate a cover letter for the most pertinent role that can be inferred.
        """ 
        self.messages.append( {"role": "user", "content": query_to_extract_info} )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.messages,
            temperature=0, # this is the degree of randomness of the model's output
        )
        response_text = (response.choices[0].message["content"])
        return response_text 
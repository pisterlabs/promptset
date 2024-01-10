from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
load_dotenv()
import os
openai_meta = {
    "keys": {
        "org": os.getenv("OPENAI_ORG_ID"),
        "api":os.getenv("OPENAI_API_KEY")
    }
}

from langchain.embeddings.openai import  OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores.pinecone import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import mindsdb_sdk
from langchain.chat_models import ChatOpenAI
chat = ChatOpenAI(temperature=0, openai_api_key=openai_meta["keys"]["api"], openai_organization=openai_meta["keys"]["org"])
from langchain.schema import AIMessage, HumanMessage, SystemMessage
server = mindsdb_sdk.connect(login=os.getenv("MINDSDB_LOGIN"), password=os.getenv('MINDSDB_PASS'))
project = server.get_project()
pinecone.init(api_key="bfad758d-abb5-409b-a2e7-ddc05f731db8", environment="us-west1-gcp-free")
embeddings = OpenAIEmbeddings(openai_api_key=openai_meta["keys"]["api"], openai_organization=openai_meta["keys"]["org"])
llm = ChatOpenAI(model_name='gpt-4', openai_api_key=openai_meta["keys"]["api"],openai_organization=openai_meta["keys"]["org"])
def Queries(index_name, namespace):
    index = pinecone.Index(index_name)
    vectordb = Pinecone.from_existing_index(index_name, embeddings, namespace=namespace)
    class Slide(BaseModel):
        script: str = Field(description="a script explaining the topic in great detail without referencing to examples")
        image_description: str = Field(description="stock image label")
        details: str = Field(description="bullet points that will be on the slides")
        code: str = Field(description="If there is code required, this field wild display it")

    class Slides(BaseModel):
        sub_topics: List[Slide] = Field(description="A JSON object representing a detailed slideshow in the format:\n{script:<a script explaining the topic in great detail without referencing to examples>,\ndetails:<bullet points that will be on the slides>\nimage_description:<image label>,\ncode:<optional, string>}")

    class QA(BaseModel):
        questions: str = Field(description="question")
        answer: str = Field(description="answer")

    class Test(BaseModel):
        test: List[QA] = Field(description="Test regarding the document")

    def store_pdf(path):
        loader = PyPDFLoader(path)
        pages = loader.load_and_split()
        Pinecone.from_documents(pages, embeddings, index_name=index_name, namespace=os.path.basename(path))

    def query_slides(question, index_name):
        parser = PydanticOutputParser(pydantic_object=Slides)
        prompt = PromptTemplate(
            input_variables=["document"],
            template="Document:\n{document}\n\nGenerate detailed slides for an educational video based on the document. Each slide should include a narration teaching the subject in detail, and a label for the image that will be shown.\n{format_instructions}\n",
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        search = vectordb.similarity_search(question)
        retriever = vectordb.as_retriever()
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        _input = prompt.format_prompt(document=search)
        output = qa(_input.to_string())
        #print(output)
        return output["result"]

    def query_flashcards(question, index_name):
        parser = PydanticOutputParser(pydantic_object=Test)
        prompt = PromptTemplate(
            input_variables=["document"],
            template="Document:\n{document}\n\nGenerate a test:\n{format_instructions}\n",
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        search = vectordb.similarity_search(question)
        retriever = vectordb.as_retriever()
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        _input = prompt.format_prompt(document=search)
        output = qa(_input.to_string())
        #print(output)
        return output["result"]


    def text_to_image(prompt):
        pred = project.query(
            f'''SELECT * 
            FROM mindsdb.dalle 
            WHERE text = "{prompt}"'''
        )
        url = pred.fetch().img_url
        return url[0]
    
    def sentence_to_prompt(sentence):
        messages = [
        SystemMessage(
            content="You are a helpful assistant that converts a sentence to keywords"
        ),
        HumanMessage(
            content="description:\n" + sentence + "\nkeywords:\n"
        ),
        ]
        response = chat(messages)
        return response.content
    
    return {
        "flashcards": lambda prompt: query_flashcards(prompt, index_name),
        "slides": lambda prompt: query_slides(prompt, index_name),
        "text_to_image": lambda prompt: text_to_image(prompt),
        "store_pdf": store_pdf,
        "sentence_to_prompt":sentence_to_prompt,
        "database": {
            "deleteAll": lambda : index.delete(deleteAll="true")
        }
    }
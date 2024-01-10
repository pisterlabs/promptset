from langchain.prompts import BaseChatPromptTemplate
from typing import Any, List, Union, Type
from langchain.agents import Tool
from langchain.schema import HumanMessage, AgentAction, AgentFinish
from langchain.agents import AgentOutputParser
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain.document_loaders import PyMuPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from pydantic import BaseModel, Field
import re
from functools import partial
from . import *

# Set up a prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    # The summary of the context
    summary: str
    
    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
            
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        print("tools: ", kwargs["tools"])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        kwargs["summary"] = self.summary
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]
    
    
class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        
        # Parse out the action and action input
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        
        # If it can't parse the output it raises an error
        # You can add your own logic here to handle errors in a different way i.e. pass to a human, give a canned response
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

class SearchInput(BaseModel):
    query: str = Field(description="should be a search query")

def db_run(db, query: str) -> str:
        return db.similarity_search_with_score(query, k=1)[0][0].page_content

class DatabaseTool:
    name = "custom_search"
    description = "Useful for when you need to answer questions about the documents given. The result will be exerpts from the documents which you will need to process further."
    arg_schema = SearchInput
    pdf_paths = []
    texts = []
    titles = []
    max_search_len = 0

    def __init__(self, pdf_paths: List[str]):
        super().__init__()
        self.pdf_paths = pdf_paths
        print("initializing database tool")
        text_splitter = SentenceTransformersTokenTextSplitter(chunk_size=1024, chunk_overlap=50)
        texts = []
        titles = []
        for pdf_path in self.pdf_paths:
            loader = PyMuPDFLoader(pdf_path)
            document = loader.load()
            titles.append(document[0])
            texts+=text_splitter.split_documents(document)
        self.max_search_len = len(texts)
        self.texts = texts
        self.titles = titles
        self.tool = Tool(name=self.name, 
                         func= partial(db_run, self.db),
                         description=self.description)
    
    @property
    def db(self):
        vectordb = Chroma.from_documents(documents=self.texts, 
                                        embedding=OpenAIEmbeddings())
        return vectordb
    
    def get_db(self):
        return self.db
    
    def get_top_k_documents(self, prompt, k):
        docs = self.db.similarity_search_with_score(prompt, k=k)
        return docs
    
    
class WriterTool:
    name = "html writer"
    description = "This will append html strings to the document you have to create. You must put valid html in the input."
    html = ""
    def __init__(self):
        print("initializing writer tool")
        self.tool = Tool(name=self.name,
                            func=self._run,
                            description=self.description)
        
    def _run(self, html: str) -> str:
        self.html += html
        return "Added to document"
    
    
    def get_html(self):
        return self.html
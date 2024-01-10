import os
import re
from typing import List, Union

# VectorStore libraries
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Weaviate

# Model libraries
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain

# Document processing libraries:
import tiktoken
from langchain.document_loaders import CSVLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Agent libraries
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate

# Library to search the web:
from langchain.utilities import GoogleSerperAPIWrapper

# API Keys
from credentials import OPENAI_API_KEY
# URL to Weaviate Cluster. Expires every 14 days.
WEAVIATE_URL = "https://prompt-cluster-8z6hr799.weaviate.network"

embeddings = OpenAIEmbeddings()

class ContextAgent():
    tools: List[Tool]
    chat_history: List[str]
    llm: ChatOpenAI()   # Brain behind all the agent operations
    agent: AgentExecutor
    
    # Contructor:
    def __init__(self):
        search = GoogleSerperAPIWrapper()        
        self.web_tool = ({
                        "Tool": Tool(
                            name = "Search",
                            func=search.run,
                            description="useful for when no other tool can provide an answer to the question."
                        ), 
                        "On": False})
    
        self.tools = []
        
        self.chat_history = []
        
        self.tokenizer = tiktoken.get_encoding('cl100k_base')
        
        self.llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name='gpt-4',
            temperature=0.8,
            verbose=True
        )
        
        self.agent = self.__create_agent()
    
    # Convert chat_history into a string:
    def __list_to_string(self):
        output = ""
        for i in self.chat_history:
            output += i
        return output
    
    # Get the length of a chunk of text in tiktokens:
    def tiktoken_len(self, text):
        tokens = self.tokenizer.encode(
            text,
            disallowed_special=()
        )
        return len(tokens)
    
    # ---------------------- LOADING AND PROCESSING NEW DATA: ---------------------------------------- 
    
    # Process the file into something an llm can use:
    def __split_data(self, file_name):
        print(f"ADFHA;ODSG;ADSKHGASD;HGADFJALSDJFKLASDJFLJASDLFKJA;DLJFASJDF;LJADSF;J !!!!{file_name}")
        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 1000,
                chunk_overlap = 200,
                length_function = self.tiktoken_len,
                separators=['\n\n', '\n', ' ', ''],
                add_start_index = True
            )
        
        base_dir = "Files"
        
        file_name = f"C:\\Users\\Alexander's Laptop\\Documents\\Computing Stuff\\LLM's\\LLMPromptSelector\\Files\\{file_name}"
        
        if file_name[-4:] == ".csv" or file_name[-4:] == ".txt":
            with open(file_name) as f:
                csv = f.read()

            documents = text_splitter.create_documents([csv])
            
        if file_name[-4:] == ".pdf":
            loader = UnstructuredPDFLoader(file_name)#, mode="elements")
            documents = loader.load()
            documents = text_splitter.split_documents(documents)
            
        if file_name[-5:] == ".html":
            loader = UnstructuredHTMLLoader(file_name)#, strategy="hi_res", mode="elements")
            documents = loader.load()
            documents = text_splitter.split_documents(documents)
            
        return documents
    
    def __create_new_tool(self, db_name, qa, description):
        self.tools.append(Tool(
            name=db_name,
            func=qa.run,
            description=(
                description
            ),
            return_direct=True
        ))
        
    # Provided file, name and description, loads the file and creates a new tool (File still needs to be in same directory):
    def load_new_data(self, uploaded_file, db_name, description):  
        
        documents = self.__split_data(uploaded_file.name)

        # create the vectorstore:
        db = Weaviate.from_documents(documents, embeddings, weaviate_url=WEAVIATE_URL, by_text=False)
        
        qa = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=db.as_retriever(),
            )
        
        self.__create_new_tool(db_name, qa, description)
        
        self.agent = self.__create_agent()  # update the agent with the new tool
    
    # ----------------------------- DEFINING AGENT ---------------------------------------
    
    # Set up a prompt template
    class CustomPromptTemplate(BaseChatPromptTemplate):
        template: str       # The template to use
        tools: List[Tool]   # The list of tools available
        
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
            # Create a list of tool names for the tools provided
            kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
            
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
            regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
            match = re.search(regex, llm_output, re.DOTALL)
            if not match:
                #return AgentAction(tool="", tool_input="", log=llm_output)
                raise ValueError(f"Could not parse LLM output: `{llm_output}`")
            action = match.group(1).strip()
            action_input = match.group(2)
            # Return the action and action input
            return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
    
    def __update_template(self):
        template = """You have access to the following tools to get relevant information/context to the question: 
        
        {tools}
        
        You should return "Final Answer: N/A" and stop all execution if you believe none of the tools are applicable or directly related to the task.
        
        You must use the following format for all responses or your response will be considered incorrect:
        
        Query: the input you must respond to
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action, should be as close to the original user query as possible
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat up to 5 times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        IF YOU DONT USE ANY TOOLS OR YOU CANT FIND THE ANSWER YOUR ONLY OUTPUT SHOULD BE: "Final Answer: N/A"

        Chat history:
        {chat_history}

        Begin!
        
        Query: {input}
        {agent_scratchpad}
        """
        
        prompt = self.CustomPromptTemplate(
            template=template,
            tools=self.tools,
            input_variables=["input", "intermediate_steps", "chat_history"]
        )
        
        return prompt
    
    def __create_agent(self):
        prompt = self.__update_template()
        
        print(f"This is the newest prompt: {prompt}")
    
        output_parser = self.CustomOutputParser()
        
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        
        tool_names = [tool.name for tool in self.tools]
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain, 
            output_parser=output_parser,
            stop=["\nObservation:"], 
            allowed_tools=tool_names
        )
        
        return AgentExecutor.from_agent_and_tools(agent=agent, tools=self.tools, verbose=True)
    
    # ------------------------- GENERATE RESPONSE ----------------------------
    
    def run(self, user_input):
        response = self.agent.run({"input" : user_input, "chat_history": self.__list_to_string()})
        return response
    
    def update_chat_history(self, user_input, response):
        self.chat_history.append(f"User query: {user_input}\n")
        self.chat_history.append(f"Agent response: {response}\n")
        # Truncate to the last 10 interactions:
        if len(self.chat_history) > 10:
            self.chat_history = self.chat_history[:-10]
    
    def toggle_search(self):
        if not self.web_tool["On"]:
            self.tools.append(self.web_tool["Tool"])    # add search to tools list
            self.web_tool["On"] = True
            self.agent = self.__create_agent()
            print("Web search is now On")
        else:
            self.tools = [tool for tool in self.tools if tool.name != "Search"]     # remove search from tools list
            self.web_tool["On"] = False
            self.agent = self.__create_agent()
            print("Web search is now Off")
    
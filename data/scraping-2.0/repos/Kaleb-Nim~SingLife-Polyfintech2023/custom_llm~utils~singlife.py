""" 
Author: Kaleb Nim
Date created: 2023-07-20
"""
import pinecone
import openai
import numpy as np
import os
from dotenv import load_dotenv
# Import time 
from time import time

# Langchain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, LLMChain
from langchain.callbacks import wandb_tracing_enabled
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from typing import Optional
from langchain.chains import SimpleSequentialChain ,SequentialChain

from langchain.chains.openai_functions import (
    create_openai_fn_chain,
    create_structured_output_chain,
)
from langchain.schema import HumanMessage, AIMessage, ChatMessage

# wandb
import wandb 

parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_TO_ENV = os.path.join(parent_directory, '.env')
print('PATH_TO_ENV: ', PATH_TO_ENV)

class Singlife:
    """
    Main Wrapper class for Main Langchin
    Assumes the .env variables are in Sn33k directory
    """

    def __init__(self):
        """
        Automatically loads the .env variables
        """
        # Load variables from the .env file
        load_dotenv('../Sn33k/.env')

        # Access the variables
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
        PINECONE_ENVIRONMENT= os.getenv("PINECONE_ENVIRONMENT")

        openai.api_key = OPENAI_API_KEY
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

        # initialize pinecone
        pinecone.init(
            api_key=PINECONE_API_KEY,  # find at app.pinecone.io
            environment=PINECONE_ENVIRONMENT,  # next to api key in console
        )

        index_name = INDEX_NAME

        embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

        # List all indexes information
        index_description = pinecone.describe_index(index_name)
        print('index_description: ', index_description)

        index = pinecone.Index(index_name) 
        index_stats_response = index.describe_index_stats()
        print('index_stats_response: ', index_stats_response)

        # Create vectorstore
        try:
            self.vectorstore = Pinecone(index, embeddings.embed_query, "text")
            print('vectorstore created succesfully')
        except Exception as e:
            raise Exception(f"Error creating vectorstore: {e}")
        
        # Default model used
        self.gpt3_model = ChatOpenAI(model_name="gpt-3.5-turbo-0613", temperature=0)
        self.gpt4_model = ChatOpenAI(model_name="gpt-4-0613", temperature=0)
        self.current_model = self.gpt3_model
        print(f'Model successfully loaded: {self.current_model}', flush=True)

        # Hard code the json schema for now since there's only one currently
        self.json_schema = self._set_json_schema()


    def generateScript(self, query: str, video_style: str = "Funny and sarcastic", model_name: str = 'gpt-3.5-turbo-0613',callbacks=None):
        """
        Main class method to generate video script from query

        Args:
            query (str): The user query to generate video script from
            video_style (str): The video prompt style to use E.g. "Funny and Energetic", "Serious and Professional", "Calm and Friendly"
            model_name (str): The model name to use for the LLMChain. Default is 'gpt-3.5-turbo-0613'
        
        Returns:
        """
        # Set the current model
        self._set_current_model(model_name=model_name)
        print(f'Current model used: {self.current_model}', flush=True)

        # Set the video prompt style
        self._set_video_prompt_style(video_style=video_style)

        # Set the json schema
        pass # TODO

        # Initialize the overall chain
        start_time = time()
        self._overallChain_init()
        print(f"Time to initialize overall chain: {start_time - time()}", flush=True)

        # Run the overall chain
        try:
            result = self.overall_chain.run(query=query)
        except Exception as e:
            raise Exception(f"Error running overall_chain: {e}")
        
        return result
    
    def _set_current_model(self, model_name: str):
        """
        Purpose:
        Set the current model to use
        """
        if model_name == 'gpt-3.5-turbo-0613':
            self.current_model = self.gpt3_model
        elif model_name == 'gpt-4-0613':
            self.current_model = self.gpt4_model
        else:
            raise Exception(f"Model name {model_name} not recognized, use 'gpt-3.5-turbo-0613' or 'gpt-4-0613'")
        
    def get_current_model(self):
        """
        Purpose:
        Get the current model to use
        """
        return self.current_model
    

    def _qaChain_init(self):
        """
        Purpose:
        Initialize the QA chain with the current model

        Returns:
            self.qa_chain to the initialized QA chain

        Error handling:
            Raises Exception if self.qa_chain cannot be initialized
        """
        # Create LLMChain
        try:
            qa_chain = RetrievalQA.from_chain_type(
            llm=self.current_model,
            chain_type="refine",
            retriever=self.vectorstore.as_retriever(),
            verbose=True,
            )
        except Exception as e:
            raise Exception(f"Error creating qa_chain: {e}")
        
        self.qa_chain = qa_chain

    def _videoChain_init(self):
        """
        Purpose:
        Initialize the video chain with the current model, video_prompt, and json_schema

        Returns:
            self.video_chain to the initialized video chain logic

        Error handling:
            Raises Exception if self.video_chain cannot be initialized
        """
        
        # Validate if self.video_prompt and self.json_schema are set
        if self.video_prompt is None:
            self._set_video_prompt_style()
            # Raise warning that video_prompt_style is being set to default
            print(f"Warning: video_prompt_style is being set to default: Funny and sarcastic", flush=True)

        if self.json_schema is None:
            self._set_json_schema()
        
        # Create video_chain
        try:
            self.video_chain = create_structured_output_chain(self.json_schema, self.current_model, self.video_prompt, verbose=True)
        except Exception as e:
            raise Exception(f"Error creating video_chain: {e}")
    
    def _overallChain_init(self):
        """
        Purpose:
        Initialize the overall chain with the current model, video_prompt, and json_schema

        Returns:
            self.overall_chain to the initialized overall chain logic

        Error handling:
            Raises Exception if self.overall_chain cannot be initialized
        """
        # Create qa_chain (RetrievalQA) --> self.qa_chain
        self._qaChain_init()
        # Create video_chain --> self.video_chain
        self._videoChain_init()
        try:
            self.overall_chain = SequentialChain(chains=[self.qa_chain, self.video_chain],input_variables=["query"])
        except Exception as e:
            raise Exception(f"Error creating overall_chain: {e}")

    def _set_video_prompt_style(self, video_style="Funny and sarcastic"):
        """
        Purpose:
        Set the video prompt style for the LLMChain, 
        Default is "Funny and sarcastic"

        Args:
            video_style (str): The video prompt style to use E.g. "Funny and Energetic", "Serious and Professional", "Calm and Friendly"
        
        Returns:
            self.video_prompt initialized to the video_style

        Error handling:
            Raises Exception video_prompt cannot be initialized
        """
        # Vaildate string input for video_style

        try:
            video_prompt = PromptTemplate(
                template="""Goal:Generate 15-30sec video script based on custom knowledge base (Information below) and user query. Two components 1.Scene assets descriptions 2.Subtitle script.\n\n------------------Custom knowledge base:------------------\n{result}\n------------------End of Custom Knowledge base.------------------\nExample Format output: dict(
                "list_of_scenes":[
                dict(
                    "scene": "Scene description1...",
                    "subtitles": [
                    "text of video subtitles1...",
                    "text of video subtitles2..."
                    ]
                ),
                dict(
                    "scene": "Scene description2...",
                    "subtitles": [
                    "text of video subtitles3...",
                    "text of video subtitles4..."
                    ]
                ),
                dict(
                    "scene": "Scene description3...",
                    "subtitles": [
                    "text of video subtitles5...",
                    "text of video subtitles6..."
                    ]
                ),
                dict(
                    "scene": "Scene description4...",
                    "subtitles": [
                    "text of video subtitles7...",
                    "text of video subtitles8..."
                    ]
                ),
                ]
            )\n\nUsing the above information in Custom knowledge base, generate a video script, video length: 15-30seconds, that addresses this user query:\n"{query}".\n\nReturn the generated video script in the style/format:"""+video_style,
                input_variables= ["result", "query"],
                validate_template=False
            )
        except Exception as e:
            raise Exception(f"Error creating video_prompt: {e}]\n\n Type video_prompt_style must be (str)")
        
        self.video_prompt= video_prompt

    def _set_json_schema(self):
        """
        TODO - add flexibility to change the json schema
        # Hard coded to one schema for now
        Purpose:
        Set the json schema for the LLMChain

        Returns:
            self.json_schema initialized to the json_schema
        """

        # Define the json schema
        json_schema = {
    "name": "format_video_script",
    "description": "Formats to a 15-30sec video script.",
    "type": "object",
    "properties": {
      "list_of_scenes": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "scene": {
              "type": "string",
              "description": "Scene description for video should be visual and general. Maximum 5 words"
            },
            "subtitles": {
              "type": "array",
              "items": {
                "type": "string",
                "description": "video subtitles script for video"
              }
            }
          },
          "required": ["scene", "subtitles"]
        }
      }
    },
    "required": ["list_of_scenes"]
  }

        self.json_schema = json_schema


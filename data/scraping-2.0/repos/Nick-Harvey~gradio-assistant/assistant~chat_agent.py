import openai
import transformers
import gradio as gr
import logging
import sys
import os
import json
import requests
import datetime
import uuid
import base64
from io import BytesIO
import time
import sys

sys.path.append("..")
from .io_utils import IO
from transformers import GenerationConfig, AutoTokenizer
from .model_utils import post_process_output, Stream, Iteratorize

# All the langchains
from langchain.tools import BaseTool
from langchain.agents import (
    Tool,
    AgentType,
    AgentExecutor,
    load_tools,
    initialize_agent,
    ZeroShotAgent,
)
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.vectorstores import DeepLake
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.chains import (
    RetrievalQA,
)
from langchain.chains.router import MultiRetrievalQAChain
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.prompts import MessagesPlaceholder
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.agents.agent_toolkits import create_retriever_tool
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.schema.output_parser import StrOutputParser

from pydantic import BaseModel


server_error_msg = (
    "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
)
load_dotenv()


class ChatAgent:
    def __init__(
        self,
        log_dir="./",
        device=None,
        io=None,
    ):
        self.log_dir = log_dir
        self.embeddings = OpenAIEmbeddings()
        self.conversation_buffer = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        self.io = io

        # Initialize ChatOpenAI instance
        self.chat = ChatOpenAI(streaming=True, temperature=0)

        # Initialize Analyst Retriever
        analyst_vs_path = "./Deeplake/snkl_helper/research_data/"
        analyst_vs = DeepLake(dataset_path=analyst_vs_path, embedding=self.embeddings)
        self.analyst_retriever = RetrievalQA.from_chain_type(
            llm=self.chat,
            chain_type="stuff",
            retriever=analyst_vs.as_retriever(),
        )

        # Initialize Snorkel Retriever
        snorkel_vs_path = "./Deeplake/snkl_helper/snorkel_data/"
        snorkel_vs = DeepLake(dataset_path=snorkel_vs_path, embedding=self.embeddings)
        self.snorkel_retriever = RetrievalQA.from_chain_type(
            llm=self.chat,
            chain_type="stuff",
            retriever=snorkel_vs.as_retriever(),
        )

    def evaluate(
        self,
        input_ids=None,
        temperature=1.0,
        top_p=0.9,
        top_k=5,
        num_beams=3,
        max_new_tokens=256,
        stream_output=True,
        length_penalty=1.0,
        no_repeat_ngram_size=2,
        do_sample=False,
        early_stopping=True,
        **kwargs
    ):
        generation_config = dict(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            do_sample=do_sample,
            early_stopping=early_stopping,
            length_penalty=length_penalty,
        )

        generate_params = {
            "input_ids": input_ids,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }
        generate_params.update(generation_config)

        if stream_output:
            # Stream the reply 1 token at a time.
            # This is based on the trick of using 'stopping_criteria' to create an iterator,
            # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.
            def generate_with_callback(callback=None, **kwargs):
                kwargs.setdefault(
                    "stopping_criteria", transformers.StoppingCriteriaList()
                )
                kwargs["stopping_criteria"].append(Stream(callback_func=callback))

            def generate_with_streaming(**kwargs):
                return Iteratorize(generate_with_callback, kwargs, callback=None)

            with generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    # new_tokens = len(output) - len(input_ids[0])
                    decoded_output = self.tokenizer.decode(output)

                    if output[-1] in [self.tokenizer.eos_token_id]:
                        break

                    yield post_process_output(decoded_output)
            return

    def predict(self, data):
        human_message = HumanMessage(
            content=data["text_input"],
            role="user",
            timestamp=datetime.datetime.utcnow(),
        )

        chat = ChatOpenAI(streaming=True, temperature=0)

        search = GoogleSearchAPIWrapper()

        tools = [
            Tool(
                name="analyst_reports",
                func=self.analyst_retriever.run,
                description="useful for all questions related to industry analysts (Gartner, IDC, etc.) reports from companies like Gartner",
            ),
            Tool(
                name="Snorkel_knowledge",
                func=self.snorkel_retriever.run,
                description="useful for all questions related to Snorkel",
            ),
            Tool(
                name="current_search",
                func=search.run,
                description="useful for questions that require up to date information",
            ),
        ]

        agent = create_conversational_retrieval_agent(self.chat, tools, verbose=True)

        try:
            response = agent({"input": data["text_input"]})

            yield (response["output"], True)

            # tokenize response then return it?

            #
            # for x in self.evaluate(response, stream_output=True, **data['generation_config']):
            #     cache = x
            #     yield (x, True)
        except ValueError as e:
            print("Caught ValueError:", e)
            yield (server_error_msg, False)
        except torch.cuda.CudaError as e:
            print("Caught torch.cuda.CudaError:", e)
            yield (server_error_msg, False)

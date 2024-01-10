from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field
from langchain.chains.base import Chain
# Langchain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, LLMChain ,LLMCheckerChain
from langchain.callbacks import wandb_tracing_enabled
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    StringPromptTemplate
)
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

from typing import Optional
from langchain.chains import SimpleSequentialChain ,SequentialChain
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent,BaseMultiActionAgent
from langchain.agents import AgentType, initialize_agent,AgentExecutor,BaseSingleActionAgent
from langchain.tools import tool
from langchain.chains.openai_functions import (
    create_openai_fn_chain,
    create_structured_output_chain,
)
from langchain.schema import HumanMessage, AIMessage, ChatMessage
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser,Agent
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, OutputParserException
import re
from langchain.agents import Tool, AgentExecutor, BaseSingleActionAgent
from langchain import OpenAI, SerpAPIWrapper

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    AsyncCallbackManagerForToolRun,
    CallbackManagerForChainRun,
    CallbackManagerForToolRun,
    Callbacks,
)

prompt = PromptTemplate(
    input_variables=['questions','user_question'],
    template = """Role: You are a question checker. Given valid questions, match the user question to one of the questions based on contextual meaning. 
    If none of the given questions match, return False

    Example 1:
    Questions:
    ---Question List "1"
    1.a) Is your loved one with dementia restless? 
    1.b) Does your loved one with dementia seem sad or upset? 
    1.c) Does your loved one with dementia seem to be looking for something?
    ---Question List "2"
    2.a) What are the psychological impacts of tube feeding? 
    2.b) tube feeding impacts
    2.c) psychological impacts of tube feeding

    User Question: 
    "potential impacts of tube feeding for my grandma"

    Your Answer:
    json[
        "isRelevantQuestion": true,
        "matched_question": "tube feeding impacts",
        "question_list_index": 2
    ]

    Example 2:
    Questions:
    ---


    ------------------------------------------------------------------------------------------------------
    Questions: 
    {questions}

    User query: 
    {user_question}

    Task:
    1. Check if the user question is relevant. releavant questions are questions the user query that has the same contextual meaning as at least one of the questions in the list of questions
    2. Given the document questions, match the user question to one of the questions based on contextual meaning.
    """
)
llm = ChatOpenAI(model_name='gpt-3.5-turbo',temperature=0)

output_schema = {
    "name": "Question Checker",
    "description": "Given valid questions, match the user question to one of the questions based on contextual meaning. If none of the given questions match, return False else return the matched question without re-phrasing. ",
    "type": "object",
    "properties": {
      "isRelevantQuestion": {
        "type": "boolean",
        "description": "Is the user question relevant? relavent questions are questions the user_question that has the same contextual meaning as at least one of the questions in the list of questions"
      },
      "matched_question":{
        "type": "string",
        "description": "Which exact question is the user question? DO NOT RE-PHRASE THE QUESTION. Return the question as is. If the user_question doesn't match any of the questions, return 'no matched'"
      },
      "question_list_index":{
        "type": "integer",
        "description": "Which question list x is the user question under? If the user_question doesn't match any of the questions, return -1"
      }
    },
    "required": ["isRelevantQuestion","matched_question","question_list_index"]
}

output_chain:LLMChain = create_structured_output_chain(llm=llm,prompt = prompt,output_schema=output_schema)


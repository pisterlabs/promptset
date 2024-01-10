from datetime import datetime

from .memory import memory
import pinecone


from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

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

patientInfo_llm = ChatOpenAI(model_name="gpt-3.5-turbo-0613", temperature=0)
patientInfoDetails_schema = {
    "name": "patientInfoKeyPoints",
    "description": "Taking in transcript of therapy session, and outputting key points of the session,including main problems faced, feelings, etc",
    "type": "object",
    "properties": {
        "MainProblems":{
            "type": "string",
            "Description": "Main problems faced by the patient, List down everything and anything that the patient is facing, no matter how small"
        },
        "AnythingRelevant":{
            "type": "string",
            "Description": "Any other relevant information, you think is useful to the theripist to aid the patient"
        },
    },# TODO: Get actual disruption Event Date, and accurate loop
    "required": ["MainProblems","AnythingRelevant"]
}

# TODO: ADD THE FUCKIGN IGIGEIGEIFNIFEANIO
patientInfoPrompt = PromptTemplate(
    template = """Role: You are a transcript extractor for a therapy session, your goal is to extract key information of the patient from the transcript between theripist and patient. Information such as Main Problems and anything u find relevant\nTranscript:\n{transcript}""",
    input_variables=["transcript"]
)
transcriptExtractorChain = create_structured_output_chain(output_schema=patientInfoDetails_schema,llm = patientInfo_llm,prompt=patientInfoPrompt)

patientInfo_llmGPT4 = ChatOpenAI(model_name="gpt-3.5-turbo-0613", temperature=0)
patientInfoPromptGPT4 = PromptTemplate(
    template = """Role: You are a transcript extractor for a therapy session, your goal is to extract key information of the patient from the transcript between theripist and patient. Information such as Main Problems and anything u find relevant\nTranscript:\n{transcript}\nFeedback for output:{feedback}""",
    input_variables=["transcript","feedback"]
)
transcriptExtractorChainGPT4 = create_structured_output_chain(output_schema=patientInfoDetails_schema,llm = patientInfo_llmGPT4,prompt=patientInfoPromptGPT4)

checkUp_llm = ChatOpenAI(model_name="gpt-3.5-turbo-0613", temperature=0)
checkUpDetails_schema = {
    "name": "eventDetails_schema",
    "description": "Generates, a questions to checkup on the patient's mental health based on the patients history",
    "type": "object",
    "properties": {
        "Question1":{
            "type": "string",
            "Description": "Perfect checkup Journal Prompt based on:\n1.History of context of the patient\nGood Journaling Prompt Examples, More about how they are feeling"
        },
        "Question2":{
            "type": "string",
            "Description": "Another relevant Journal Prompt question, very different froom Question1"
        },
    },# TODO: Get actual disruption Event Date, and accurate loop
    "required": ["Question1","Question2"]
}

checkUpPrompt = PromptTemplate(
    template = """Role:You are a Therapy Journal Prompter, your goal is get the patient to Journel their thoughts/feelings by asking them relevant questions to their issues. Craft the perfect checkup Question based on:\n1.History of context of the patient\n2. Example questions of a good Journaling Prompt.\n\nGood Journaling Prompt Examples:\n1.What kind of goals and objectives would u want to set, related to this problem X or challenge X?\n2.How do you think you should go about prioritizing and organize my thoughts and ideas to effectively solve this problem or challenge?\n3.What did I do today that I am proud of?\n\n Patient History Context:1. Main Problems: {MainProblems}\n2. Anything Other Relevant information: {AnythingRelevant}\n\nUsing the above information, craft the perfect Journal Prompt based on the patient's history and the example questions of a good Journaling Prompt.Must output Json""",
    input_variables=["MainProblems","AnythingRelevant"]
)
checkUpChain = create_structured_output_chain(output_schema=checkUpDetails_schema,llm = checkUp_llm,prompt=checkUpPrompt)

checkUp_llmGPT4 = ChatOpenAI(model_name="gpt-4-0613", temperature=0)
checkUpPromptGPT4 = PromptTemplate(
    template = """Role:You are a Theripst checking up on a patient daily, your goal is get the patient to Journel their thoughts/feelings by asking them relevant questions. Craft the perfect checkup Question based on:\n1.History of context of the patient\n2. Example questions of a good Journaling Prompt.\n\nGood Journaling Prompt Examples:\n1.What kind of goals and objectives would u want to set, related to this problem X or challenge X?\n2.How do you think you should go about prioritizing and organize my thoughts and ideas to effectively solve this problem or challenge?\n3.What did I do today that I am proud of?\n\n Patient History Context:1. Main Problems: {MainProblems}\n2. Anything Other Relevant information: {AnythingRelevant}\n\nUsing the above information, craft the perfect Journal Prompt based on the patient's history and the example questions of a good Journaling Prompt.Must output Json.\nFeedback:{feedback}""",
    input_variables=["MainProblems","AnythingRelevant","feedback"]
)
checkUpChainGPT4 = create_structured_output_chain(output_schema=checkUpDetails_schema,llm = checkUp_llmGPT4,prompt=checkUpPromptGPT4)


followUpCheckUpAdvice_llm = ChatOpenAI(model_name="gpt-3.5-turbo-0613", temperature=0)

followUpCheckUpAdviceDetails_schema = {
    "name": "followUpCheckUpAdviceDetails_schema",
    "description": "Generates, a follow up checkup questions and advice to the patient's mental health based on the patients history",
    "type": "object",
    "properties": {
        "Advice1":{
            "type": "string",
            "Description": "Perfect follow up advice based on:\n1.History of context of the patient\n and current journal reflection of the patient\nGood Journaling Prompt Examples"
        },
        "Advice2":{
            "type": "string",
            "Description": "Another relevant follow up checkup question, very different froom Advice1"
        },
}
}

followUpCheckUpAdvicePrompt = PromptTemplate(

    template = """Role:You are a Theripst checking up on a patient daily, your goal is to give the patient advice based on their journal reflection. Craft the perfect advice based on:\n1.History of context of the patient:\nMain Problems: {MainProblems}\nAnything Other Relevant information: {AnythingRelevant}\n2.Journal Prompts asked previously:Question1:{Question1}\nQuestion2:{Question2}3. Current journal reflection of the patient:{PatientJournalReflection}3.\n\nUsing the above information, craft the perfect Journal Prompt based on the 1.patient's history context,2. previous journal prompts asked, and 3. current journal reflection of the patient.""",
    input_variables=["MainProblems","AnythingRelevant","Question1","Question2","PatientJournalReflection"]

)

followUpCheckUpAdviceChain = create_structured_output_chain(output_schema=followUpCheckUpAdviceDetails_schema,llm = followUpCheckUpAdvice_llm,prompt=followUpCheckUpAdvicePrompt)

followUpCheckUpAdvice_llmGPT4 = ChatOpenAI(model_name="gpt-4-0613", temperature=0)
followUpCheckUpAdvicePromptGPT4 = PromptTemplate(
    
        template = """Role:You are a Theripst checking up on a patient daily, your goal is to give the patient advice based on their journal reflection. Craft the perfect advice based on:\n1.History of context of the patient:\nMain Problems: {MainProblems}\nAnything Other Relevant information: {AnythingRelevant}\n2.Journal Prompts asked previously:Question1:{Question1}\nQuestion2:{Question2}3. Current journal reflection of the patient:{PatientJournalReflection}3.\n\nUsing the above information, craft the perfect Journal Prompt based on the 1.patient's history context,2. previous journal prompts asked, and 3. current journal reflection of the patient.\nFeedback:{feedback}""",
        input_variables=["MainProblems","AnythingRelevant","Question1","Question2","PatientJournalReflection","feedback"]
    
    )
followUpCheckUpAdviceChainGPT4 = create_structured_output_chain(output_schema=followUpCheckUpAdviceDetails_schema,llm = followUpCheckUpAdvice_llmGPT4,prompt=followUpCheckUpAdvicePromptGPT4)
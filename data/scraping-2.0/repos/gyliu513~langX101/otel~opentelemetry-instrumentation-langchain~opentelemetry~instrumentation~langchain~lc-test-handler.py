"""Langchain BaseHandler instrumentation"""
import logging
from typing import Collection

from opentelemetry.trace import get_tracer
from opentelemetry.instrumentation.langchain.version import __version__
from opentelemetry.semconv.ai import TraceloopSpanKindValues
from otel_lib.instrumentor import LangChainHandlerInstrumentor


logger = logging.getLogger(__name__)

_instruments = ("langchain >= 0.0.200",)


from dotenv import load_dotenv, find_dotenv
import os
load_dotenv(find_dotenv())

os.environ['OTEL_EXPORTER_OTLP_INSECURE'] = 'True'

import sys

from opentelemetry import trace
# from opentelemetry.instrumentation.wsgi import collect_request_attributes
from opentelemetry.propagate import extract
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter


from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)
from opentelemetry.trace import (
    SpanKind,
    get_tracer_provider,
    set_tracer_provider,
)

tracer_provider = TracerProvider(
    resource=Resource.create({'service.name': os.environ["SVC_NAME"]}),
)

# Create an OTLP Span Exporter
otlp_exporter = OTLPSpanExporter(
    endpoint=os.environ["OTLP_EXPORTER"]+":4317",  # Replace with your OTLP endpoint URL
)

# Add the exporter to the TracerProvider
# tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))  # Add any span processors you need
tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

# Register the TracerProvider
trace.set_tracer_provider(tracer_provider)

LangChainHandlerInstrumentor().instrument(tracer_provider=tracer_provider)

import os
import openai

from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain

openai.api_key = os.getenv("OPENAI_API_KEY")

def parsePromptTemplate(prompt):
    for item in prompt.messages:
        print(f"type:{type(item)}")
        if hasattr(item, "content"):
            print(item.content)
        elif hasattr(item, "prompt"):
            print(type(item.prompt))
            print(f"item.prompt.input_variables:{item.prompt.input_variables}")
            print(f"item.prompt.template: {item.prompt.template}")
        print(f"item:{item}")
        
def printClassDetails(c):
    attrs = vars(c)    
    for key, value in attrs.items():
        print(f"{key}: {value}")
    

def langchain_app():
    chat = ChatOpenAI(temperature=0, max_tokens=30)

    # messages = [
    #     SystemMessage(content="You are a calculator"),
    #     HumanMessage(content="tell me the result of 1+1=")
    # ]
    # print(chat(messages)) 

    first_prompt_messages = [
        SystemMessage(content="You are a funny sarcastic nerd."),
        HumanMessage(content="Tell me a joke about OpenTelemetry."),
    ]
    first_prompt_template = ChatPromptTemplate.from_messages(first_prompt_messages)
    first_chain = LLMChain(llm=chat, prompt=first_prompt_template, output_key="joke")

    second_prompt_messages = [
        SystemMessage(content="You are an Elf."),
        HumanMessagePromptTemplate.from_template(
            "Translate the joke below into Sindarin language:\n {joke}"
        ),
    ]
    second_prompt_template = ChatPromptTemplate.from_messages(second_prompt_messages)
    second_chain = LLMChain(llm=chat, prompt=second_prompt_template)

    workflow = SequentialChain(chains=[first_chain, second_chain], input_variables=[])
    workflow({})


langchain_app()
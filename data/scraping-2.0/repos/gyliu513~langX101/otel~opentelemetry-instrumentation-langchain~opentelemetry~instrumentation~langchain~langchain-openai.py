"""OpenTelemetry Langchain instrumentation"""
import logging
from typing import Collection
from wrapt import wrap_function_wrapper

from opentelemetry.trace import get_tracer

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap

from opentelemetry.instrumentation.langchain.task_wrapper import task_wrapper
from opentelemetry.instrumentation.langchain.workflow_wrapper import workflow_wrapper
from opentelemetry.instrumentation.langchain.version import __version__

from opentelemetry.semconv.ai import TraceloopSpanKindValues

logger = logging.getLogger(__name__)

_instruments = ("langchain >= 0.0.200",)

WRAPPED_METHODS = [
    {
        "package": "langchain.chains.base",
        "object": "Chain",
        "method": "__call__",
        "wrapper": task_wrapper,
    },
    {
        "package": "langchain.chains.base",
        "object": "Chain",
        "method": "acall",
        "wrapper": task_wrapper,
    },
    {
        "package": "langchain.chains",
        "object": "SequentialChain",
        "method": "__call__",
        "span_name": "langchain.workflow",
        "wrapper": workflow_wrapper,
    },
    {
        "package": "langchain.chains",
        "object": "SequentialChain",
        "method": "acall",
        "span_name": "langchain.workflow",
        "wrapper": workflow_wrapper,
    },
    {
        "package": "langchain.agents",
        "object": "AgentExecutor",
        "method": "_call",
        "span_name": "langchain.agent",
        "kind": TraceloopSpanKindValues.AGENT.value,
        "wrapper": workflow_wrapper,
    },
    {
        "package": "langchain.tools",
        "object": "Tool",
        "method": "_run",
        "span_name": "langchain.tool",
        "kind": TraceloopSpanKindValues.TOOL.value,
        "wrapper": task_wrapper,
    },
    {
        "package": "langchain.chains",
        "object": "RetrievalQA",
        "method": "__call__",
        "span_name": "retrieval_qa.workflow",
        "wrapper": workflow_wrapper,
    },
    {
        "package": "langchain.chains",
        "object": "RetrievalQA",
        "method": "acall",
        "span_name": "retrieval_qa.workflow",
        "wrapper": workflow_wrapper,
    },
]


class LangchainInstrumentor(BaseInstrumentor):
    """An instrumentor for Langchain SDK."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)
        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            wrapper = wrapped_method.get("wrapper")
            wrap_function_wrapper(
                wrap_package,
                f"{wrap_object}.{wrap_method}" if wrap_object else wrap_method,
                wrapper(tracer, wrapped_method),
            )

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            unwrap(
                f"{wrap_package}.{wrap_object}" if wrap_object else wrap_package,
                wrap_method,
            )

'''
test program
'''

from dotenv import load_dotenv
import os
load_dotenv()

os.environ['OTEL_EXPORTER_OTLP_INSECURE'] = 'True'

import sys

from opentelemetry import trace
from opentelemetry.instrumentation.wsgi import collect_request_attributes
from opentelemetry.propagate import extract
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from opentelemetry.instrumentation.langchain import LangchainInstrumentor

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

LangchainInstrumentor().instrument(tracer_provider=tracer_provider)

import os
import openai

from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain

from traceloop.sdk import Traceloop

openai.api_key = os.getenv("OPENAI_API_KEY")

def langchain_app():
    chat = ChatOpenAI(temperature=0)

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
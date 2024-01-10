"""Langchain BaseHandler instrumentation"""
import logging
import time
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

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter as OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter as OTLPMetricExporterHTTP
from opentelemetry.metrics import (
    CallbackOptions,
    Observation,
    get_meter_provider,
    set_meter_provider,
)

resource=Resource.create({'service.name': os.environ["SVC_NAME"]})
span_endpoint=os.environ["OTLP_EXPORTER"]+":4317"         # Replace with your OTLP endpoint URL
metric_endpoint=os.environ["OTLP_EXPORTER"]+":4317"       # Replace with your Metric endpoint URL
metric_http_endpoint=os.environ["METRIC_EXPORTER_HTTP_TESTING2"]

# testing metrics endpoint 
# metric_endpoint=os.environ["METRIC_EXPORTER_TESTING"]
# metric_endpoint=os.environ["OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"]

tracer_provider = TracerProvider(
    resource = resource,
)

# Create an OTLP Span Exporter
otlp_exporter = OTLPSpanExporter(
    endpoint=span_endpoint,
)

# Add the exporter to the TracerProvider
# tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))  # Add any span processors you need
tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

# Register the trace provider
trace.set_tracer_provider(tracer_provider)

reader = PeriodicExportingMetricReader(
    # OTLPMetricExporter(endpoint=metric_endpoint)
    OTLPMetricExporterHTTP(endpoint=metric_http_endpoint)
)

# Metrics console output
# console_reader = PeriodicExportingMetricReader(ConsoleMetricExporter())

metric_provider = MeterProvider(resource=resource, metric_readers=[reader])
# Register the metric provide
metrics.set_meter_provider(metric_provider)


LangChainHandlerInstrumentor().instrument(tracer_provider=tracer_provider, metric_provider=metric_provider)

os.environ['OTEL_EXPORTER_OTLP_INSECURE'] = 'True'
os.environ["WATSONX_APIKEY"] = os.getenv("IAM_API_KEY")

# from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as WatsonMLGenParams

# watson_ml_parameters = {
#     WatsonMLGenParams.DECODING_METHOD: "sample",
#     WatsonMLGenParams.MAX_NEW_TOKENS: 30,
#     WatsonMLGenParams.MIN_NEW_TOKENS: 1,
#     WatsonMLGenParams.TEMPERATURE: 0.5,
#     WatsonMLGenParams.TOP_K: 50,
#     WatsonMLGenParams.TOP_P: 1,
# }

# from langchain.llms import WatsonxLLM

# watsonx_ml_llm = WatsonxLLM(
#     model_id="google/flan-ul2",
#     url="https://us-south.ml.cloud.ibm.com",
#     project_id=os.getenv("PROJECT_ID"),
#     params=watson_ml_parameters,
# )

from genai.extensions.langchain import LangChainInterface
from genai.schemas import GenerateParams as GenaiGenerateParams
from genai.credentials import Credentials
from otel_lib.country_name import RandomCountryName

api_key = os.getenv("IBM_GENAI_KEY", None) 
api_url = "https://bam-api.res.ibm.com"
creds = Credentials(api_key, api_endpoint=api_url)

genai_parameters = GenaiGenerateParams(
    decoding_method="sample",  # Literal['greedy', 'sample']
    max_new_tokens=300,
    min_new_tokens=10,
    top_p=1,
    top_k=50,
    temperature=0.05,
    time_limit=30000,
    # length_penalty={"decay_factor": 2.5, "start_index": 5},
    # repetition_penalty=1.2,
    truncate_input_tokens=2048,
    # random_seed=33,
    stop_sequences=["fail", "stop1"],
    return_options={
        "input_text": True, 
        "generated_tokens": True, 
        "input_tokens": True, 
        "token_logprobs": True, 
        "token_ranks": False, 
        "top_n_tokens": False
        },
)

watsonx_genai_llm = LangChainInterface(
    # model="google/flan-t5-xxl", 
    # model="meta-llama/llama-2-70b", 
    model = "ibm/granite-13b-chat-v1",
    params=genai_parameters, 
    credentials=creds
)

from langchain.prompts import PromptTemplate
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI


def langchain_watson_genai_llm_chain():
    from langchain.schema import SystemMessage, HumanMessage
    from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
    from langchain.chains import LLMChain, SequentialChain

    openai_llm = OpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], temperature=0.1)
    
    first_prompt_messages = [
        SystemMessage(content="answer the question with very short answer, as short as you can."),
        # HumanMessage(content=f"tell me what is the most famous tourist attraction in the capital city of {RandomCountryName()}?"),
        HumanMessage(content=f"tell me what is the most famous dish in {RandomCountryName()}?"),
    ]
    first_prompt_template = ChatPromptTemplate.from_messages(first_prompt_messages)
    first_chain = LLMChain(llm=watsonx_genai_llm, prompt=first_prompt_template, output_key="target")

    second_prompt_messages = [
        SystemMessage(content="answer the question with very brief answer."),
        # HumanMessagePromptTemplate.from_template("how to get to {target} from the nearest airport by public transportation?\n "),
        HumanMessagePromptTemplate.from_template("pls provide the recipe for dish {target}\n "),
    ]
    second_prompt_template = ChatPromptTemplate.from_messages(second_prompt_messages)
    second_chain = LLMChain(llm=watsonx_genai_llm, prompt=second_prompt_template)

    workflow = SequentialChain(chains=[first_chain, second_chain], input_variables=[])
    print(workflow({}))
    

def langchain_serpapi_math_agent():
    openai_llm = OpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], temperature=0.1)

    tools = load_tools(["serpapi", "llm-math"], llm=watsonx_genai_llm)

    agent = initialize_agent(
        tools, openai_llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    # agent.run("My monthly salary is 10000 KES, if i work for 10 months. How much is my total salary in USD in those 10 months.")
    print(agent.run("a pair of shoes sale price 300 CNY and a beautiful pocket knife price at 50 USD, how much in USD if I want them both?"))

def langchain_chat_memory_agent():
    from langchain.memory import ConversationBufferMemory
    
    memory = ConversationBufferMemory(memory_key="chat_history")
    
    tools = load_tools(["serpapi", "llm-math"], llm=watsonx_genai_llm)

    agent = initialize_agent(tools, watsonx_genai_llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)
    print(agent.run(f"what is the capital city of {RandomCountryName()}?"))
    print(agent.run("what is the most famous dish of this city?"))
    print(agent.run("pls provide a receipe for this dish"))



# langchain_serpapi_math_agent()

# langchain_chat_memory_agent()

langchain_watson_genai_llm_chain()

# interval = 180 
# count = 100
# while count > 0:
#     count -= 1
#     langchain_watson_genai_llm_chain()
#     time.sleep(interval)

metric_provider.force_flush()
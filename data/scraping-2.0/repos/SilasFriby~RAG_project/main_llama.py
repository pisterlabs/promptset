import openai
import os
from dotenv import load_dotenv
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
    PromptHelper,
    set_global_service_context,
)
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms import OpenAI
from llama_index.text_splitter import SentenceSplitter
from llama_index.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
)
from llama_index.llms import HuggingFaceLLM, HuggingFaceInferenceAPI
from llama_index.llms.huggingface import chat_messages_to_conversational_kwargs
import torch
from llama_index.prompts import PromptTemplate
import accelerate
from llama_index.llms.types import ChatMessage

# Initialize variables
documents_dir = "data/statements_txt_files"
llm_model = "gpt-3.5-turbo-0613"
llm_response_max_tokens = 256
llm_temp = 0
chunk_size = 1024
chunk_overlap = 0
paragraph_separator = "\n\n"
system_prompt = "Hello, I am a financial analyst. My expertise is answering questions about financial statements."


# Load environment variables from .env file
load_dotenv()

# Hugging Face Token
HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

remotely_run = HuggingFaceInferenceAPI(
    model_name="berkeley-nest/Starling-LM-7B-alpha", 
    token=HF_TOKEN
)

message = [ChatMessage(
    role="user",  # Role can be 'user' or 'system'
    content="What is the capital of France?"  # The actual message content
)]

response = remotely_run.chat(message)
print(response)

# Get the OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set the OpenAI API key
openai.api_key = openai_api_key

# LLM
# llm = OpenAI(model=llm_model, temperature=llm_temp, max_tokens=llm_response_max_tokens)

query_wrapper_prompt = PromptTemplate(
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{query_str}\n\n### Response:"
)

llm = HuggingFaceLLM(
    context_window=2048,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0, "do_sample": False},
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="Writer/camel-5b-hf",
    model_name="Writer/camel-5b-hf",
    device_map="auto",
    tokenizer_kwargs={"max_length": 2048},
    # uncomment this if using CUDA to reduce memory usage
    model_kwargs={"torch_dtype": torch.float16}
)


# Read the documents from the directory
reader = SimpleDirectoryReader(input_dir=documents_dir, filename_as_id=True)
documents = reader.load_data()

# Chunking
text_splitter = SentenceSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    paragraph_separator=paragraph_separator,
)

# Meta data extractors
meta_data_extractors = [
    TitleExtractor(nodes=1, llm=llm),
    # QuestionsAnsweredExtractor(questions=3, llm=llm),
    # SummaryExtractor(summaries=["prev", "self"], llm=llm),
    # KeywordExtractor(keywords=10, llm=llm),
]

# Embedding
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Prompt Helper
prompt_helper = PromptHelper()

# Service Context - a bundle of commonly used resources used during the indexing and querying stage in a LlamaIndex pipeline
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
    text_splitter=text_splitter,
    prompt_helper=prompt_helper,
    system_prompt=system_prompt,
    transformations=[text_splitter] + meta_data_extractors,
)

# # Set service context as the global default that applies to the entire LlamaIndex pipeline
# set_global_service_context(service_context)

# Indexing
index = VectorStoreIndex.from_documents(
    documents, service_context=service_context, show_progress=True
)

# Create chat engine that uses the index
chat_engine = index.as_chat_engine(chat_mode="openai", verbose=True)

response = chat_engine.chat(
    "What was the revenues for UP Fintech and Top Strike?",
    tool_choice="query_engine_tool",
)
print(response)

response = chat_engine.chat(
    "What was the revenue for Top Strike?",
    tool_choice="query_engine_tool",
)
print(response)

# LLM from HugginFace? At least for meta-data extraction
# Can we see that meta data is being used? E.g is it stored in the vector store index?
# Can we save vector index to disk?
# NEXT UP!! SUBQUERY OR STEP-BACK PROMPTING IN CASE WE WISH TO COMPARE COMPANIES

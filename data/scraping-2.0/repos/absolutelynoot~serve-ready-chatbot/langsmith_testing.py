import os
from dotenv import load_dotenv
from langsmith import Client
from langchain.chat_models import ChatOpenAI
from langchain.smith import RunEvalConfig, run_on_dataset
from langchain import smith
import langsmith

# Load the API keys from the .env file

load_dotenv('./.env')
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ["LANGCHAIN_PROJECT"] = "Serve Ready Chatbot LangChain Testing"

client = langsmith.Client()

llm_name = "gpt-3.5-turbo-16k"
llm = ChatOpenAI(model_name=llm_name, temperature=0)

# Load the csv file onto langsmith

csv_path = "./docs/AIH-G1 Project Management & UAT - UAT.csv"
input_keys = ["Test Data"]
output_keys = ["Generated Result"]

dataset = client.upload_csv(
    csv_file=csv_path,
    input_keys=input_keys,
    output_keys=output_keys,
    name="Serve Ready Chatbot LangChain Dataset",
    description="Dataset for testing Serve Ready Chatbot",
    data_type="kv",
)

# Langsmith QA evaluation, 3 metrics context_qa, qa, cot_qa

evaluation_config = smith.RunEvalConfig(
    evaluators=[
        "cot_qa",
        "qa",
        "context_qa"
    ]
)

client.run_on_dataset(
    dataset_name="Serve Ready Chatbot LangChain Dataset",
    llm_or_chain_factory=llm,
    evaluation=evaluation_config,
    project_name="Serve Ready Chatbot Testing",
    concurrency_level=5,
    verbose=True,
)

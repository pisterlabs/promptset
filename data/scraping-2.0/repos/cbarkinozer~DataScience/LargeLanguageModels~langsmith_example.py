# This notebook is a demonstration of how to run HumanEval while taking advantage of LangSmith's visibility and tracing features.

# To use it:
# 1. Update the settings and API keys below.
# 2. Run the notebook.
# 3. View results in LangSmith.

# Dependencies

!pip install -q langchain langsmith codechain openai human-eval

# API keys

import os

os.environ["OPENAI_API_KEY"] = "sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
os.environ["LANGCHAIN_API_KEY"] = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

# Dataset settings

description = "HumanEval dataset"

dataset_name, max_problems = "humaneval-small", 3
#dataset_name, max_problems = "humaneval-all", False

repetitions_per_problem = 5

# LLM settings

model_name = "gpt-4"
temperature = 0.2

# Langsmith settings

os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# LangSmith client

import langsmith

client = langsmith.Client()
print(client)

# Dataset creation

from human_eval.data import read_problems

# Get the HumanEval dataset up to max_problems.
problems = read_problems()
if max_problems:
  problems = {key: problems[key] for key in list(problems.keys())[:max_problems]}

# If the dataset is new, update it to the LangSmith server.
if dataset_name not in set([dataset.name for dataset in client.list_datasets()]):
  dataset = client.create_dataset(dataset_name, description=description)
  for key, value in problems.items():
      client.create_example(
          inputs={
              "prompt": value["prompt"],
              "task_id": key
              },
          outputs={
              "canonical_solution": value["canonical_solution"],
              },
          dataset_id=dataset.id
      )

# Generation and evaluation

from codechain.generation import HumanEvalChain, CompleteCodeChain
from codechain.evaluation import HumanEvalEvaluator

from langchain.chat_models import ChatOpenAI
from langchain.smith import arun_on_dataset, RunEvalConfig

# Factory for the generation chain
def chain_factory():
    """Create a code generation chain."""
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    return HumanEvalChain.from_llm(llm)

# Evaluator configuration
evaluation = RunEvalConfig(
    custom_evaluators=[HumanEvalEvaluator()],
    input_key="task_id"
    )

# Run all generations and evaluations
chain_results = await arun_on_dataset(
    client=client,
    dataset_name=dataset_name,
    num_repetitions=repetitions_per_problem,
    concurrency_level=5,
    llm_or_chain_factory=chain_factory,
    evaluation=evaluation,
    tags=["HumanEval"],
    verbose=True
)
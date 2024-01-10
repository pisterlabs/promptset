import langsmith

from langchain import chat_models, smith
from langchain.chat_models import ChatOllama

# Replace with the chat model you want to test
model = ChatOllama(model="mistral")

# Define the evaluators to apply
eval_config = smith.RunEvalConfig(
    evaluators=[
        "cot_qa",
        smith.RunEvalConfig.LabeledCriteria("relevance"),
        smith.RunEvalConfig.LabeledCriteria("coherence")
    ],
    custom_evaluators=[],
    eval_llm=model
)

client = langsmith.Client()
chain_results = client.run_on_dataset(
    dataset_name="gbif-api-agents-results",
    llm_or_chain_factory=model,
    evaluation=eval_config,
    project_name="gbif-api-agents-results",
    concurrency_level=5,
    verbose=True,
)
import langsmith

from langchain import chat_models

# Replace with the chat model you want to test
my_llm = chat_models.ChatOpenAI(temperature=0)

client = langsmith.Client()
chain_results = client.run_on_dataset(
    dataset_name="提取收货地址",
    llm_or_chain_factory=my_llm,
    project_name="test-minty-mapping-48",
    verbose=True,
)
import os

from chainforge.providers import provider
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.you import YouRetriever

# Set up environment variables for API keys
os.environ["YDC_API_KEY"] = os.getenv("YOU_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize You.com's LangChain retriever and set the model
yr = YouRetriever()
model = "gpt-3.5-turbo-16k"


def get_genwebsearch_response(prompt: str) -> str:
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model=model), chain_type="stuff", retriever=yr
    )
    return qa.run(prompt)


@provider(
    name="You.com (gpt-3.5-turbo)",
    emoji="🌈",
    models=[model],
    rate_limit="sequential",
    settings_schema={},
)
def YouOpenAIProvider(prompt: str, **kwargs) -> str:
    """ChainForge custom provider that uses You.com & Open AI for generating responses."""
    return get_genwebsearch_response(prompt)


# Test the functioning (outside of ChainForge's provider)
if __name__ == "__main__":
    test_prompt = "Explain the latest news this week in generative web search."
    result = get_genwebsearch_response(test_prompt)
    print("Test Prompt:", test_prompt)
    print("Provider Response:", result)

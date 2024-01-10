from pydantic import BaseModel, Field
from langchain.retrieval.qa import RetrievalQAChain
from langchain.conversation.agent import ConversationAgent

# Define the custom response schema
class CustomResponseSchema(BaseModel):
    """An answer to the question being asked, with sources."""
    answer: str = Field(..., description="Answer to the question that was asked")
    sources: List[str] = Field(
        ..., description="List of sources used to answer the question"
    )

def create_agent(doc_retrieval_chain: RetrievalQAChain, 
                 repo_retrieval_chain: RetrievalQAChain) -> ConversationAgent:
    """
    Function to create a conversation agent that uses the RetrievalQaChains 
    to provide relevant information based on the user's query.
    """

    # Create a conversation agent with the retrieval chains
    agent = ConversationAgent([doc_retrieval_chain, repo_retrieval_chain])

    return agent

# Load the retrieval chains from the other modules
from generated.retriever_module import doc_retrieval_chain, repo_retrieval_chain

# Create the agent
agent = create_agent(doc_retrieval_chain, repo_retrieval_chain)

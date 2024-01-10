from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub


def create_conversation_chain(model_type, vectorstore):
    """
    This function creates a conversation chain for conversation retrieval

    Inputs:
        model_type - huggingface or openai
        vectorstore - The vector database containing the embeddings of the text
    Outputs:
        conversation_chain
    """

    if model_type == "Open AI":
        # Load OpenAI conversational model
        model = ChatOpenAI()
    else: 
        # Load hugging face model
        model = HuggingFaceHub(repo_id = "google/flan-t5-xxl", model_kwargs = {"temperature": 0.5, "max_length": 512}) 

    # memory buffer is used to store and retrieve previous conversation history. Used to maintain context and continuity
    memory = ConversationBufferMemory(memory_key = "chat_history", return_messages = True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = model,
        retriever = vectorstore.as_retriever(),
        memory = memory
    )
    return conversation_chain


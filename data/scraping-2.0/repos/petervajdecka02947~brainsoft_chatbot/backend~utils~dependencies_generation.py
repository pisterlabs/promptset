from config import settings
import pinecone
import openai
from utils.callback_handler_agent import *
from utils.callback_handler_chain import *
from utils.error_handler import *
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone
from langchain.tools import DuckDuckGoSearchRun, Tool
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


# Add your utility functions here. This is a placeholder.
def update_openai_api_key(new_key: str, model: str):
    """
    Updates the OpenAI API key and model in the application settings.

    Args:
        new_key (str): The new OpenAI API key.
        model (str): The model to be used with the new API key.

    Returns:
        object: The updated application settings.

    This function updates the OpenAI API key and model, and performs a test call to OpenAI's chat API to verify the new key.
    If the test call fails, it raises an UpdateError.

    Raises:
        UpdateError: If the test call to OpenAI's API fails.
    """

    try:
        openai.api_key = new_key

        # Making a simple test call to OpenAI's chat API
        openai.ChatCompletion.create(
            model=model,  # Specify the chat model here
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, who are you?"},
            ],
        )

        settings.OPENAI_API_KEY = new_key
        settings.LLM_NAME = model

        # If the call is successful, the key is working
        return settings

    except openai.error.OpenAIError as e:
        raise UpdateError(
            f"Failed to update API key and model due to OpenAI API error: {e}", 400
        )


def setup_conversational_chain(settings: object):
    """
    Initializes the conversational chain with various tools and configurations.

    Args:
        settings (object): Application settings containing configuration details.

    Returns:
        tuple: A tuple containing the agent, retriever, and language model chain.

    This function sets up the conversational agent with various tools (like retrievers and search functions)
    and configures the language model chain. It handles the initialization of the database, vector database,
    language model, and other components required for the conversational chain.

    Raises:
        UpdateError: If there is an error during the initialization of any component.
    """
    global agent
    global retriever
    global llm

    tools = []

    # Initialize database

    try:
        pinecone.init(
            api_key=settings.PINECONE_API_KEY, environment=settings.PINECONE_ENV
        )

        embeddings_model = OpenAIEmbeddings(
            model=settings.EMBEDDING_NAME, openai_api_key=settings.OPENAI_API_KEY
        )

        vectordb = Pinecone.from_existing_index(settings.INDEX_NAME, embeddings_model)

    except Exception as e:
        raise UpdateError(f"Error during initialization of vector database: {e}", 401)

    # Initialize database LLM model

    try:
        llm = ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            model_name=settings.LLM_NAME,
            temperature=0,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        )
    except Exception as e:
        raise UpdateError(f"Error during initialization of LLM: {e}", 402)
    # Prepare retriever

    try:
        retriever = vectordb.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.1, "k": 1},
        )
    except Exception as e:
        raise UpdateError(f"Error during initialization of retriever: {e}", 403)

    # Initialize tools

    try:
        tool_retrieve = create_retriever_tool(
            retriever,
            "Doc_search",
            "Searches and returns documents regarding the IBM Generative AI Python SDK documentation",
        )

        search = DuckDuckGoSearchRun()
        search_tool = Tool(
            name="DuckDuckGo",
            func=search,  # .run
            description="This tool is used when you need to do a search on the internet to find information that another tool Doc_search can't find.",
        )

        tools.append(tool_retrieve)
        tools.append(search_tool)

        #        Initialize tools
        agent = initialize_agent(
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            tools=tools,
            llm=llm,
            verbose=True,
            max_iterations=10,
            early_stopping_method="generate",
            # memory=memory,
            return_intermediate_steps=False,
        )

    except Exception as e:
        raise UpdateError(f"Error during initialization of toolls an agent: {e}", 404)
    # Set template for history input

    try:
        new_string = "\n\n\nCHAT HISTORY AND USER'S INPUT\n-----------------------------\nHere is the Chat history followed by user's input"
        old_string = (
            "\n```\n\nUSER'S INPUT\n--------------------\nHere is the user's input"
        )
        template_new = agent.agent.llm_chain.prompt.messages[2].prompt.template.replace(
            old_string, new_string
        )

        agent.agent.llm_chain.prompt.messages[2].prompt.template = template_new

    except Exception as e:
        raise UpdateError(f"Unable to set custom template: {e}", 405)

    try:
        template = (
            """Answer the following question based on the context provided. Generate a precise and accurate answer, ensuring it is fully"""
            """supported by the provided context. Make sure you never include symbol " in your output.
        Context: {context}
        Question:{input}
        Answer: """
        )

        prompt = PromptTemplate(template=template, input_variables=["input", "context"])

        llm_chain = LLMChain(prompt=prompt, llm=llm)

    except Exception as e:
        raise UpdateError(f"Unable to set OpenAI chain: {e}", 406)

    return agent, retriever, llm_chain


# Retrieve the relevant document


async def get_source(retriever_obj: object, query: str):
    """
    Retrieves the relevant document source based on a given query.

    Args:
        retriever_obj (object): The retriever object to be used for document retrieval.
        query (str): The query string for which relevant document source is needed.

    Returns:
        str: The source of the relevant document if found, otherwise a default value.

    This function queries the retriever object for relevant documents based on the input query.
    It returns the source of the first relevant document if found, or a default value ('DuckDuckGo' or 'No ibm related source found') otherwise.

    Note:
        The function returns 'DuckDuckGo' if no documents are found or if an exception occurs.
    """
    try:
        docs = await retriever_obj.aget_relevant_documents(query)
    except Exception as e:
        return "Not retrieved"
    if docs == []:
        return "Not retrieved"
    else:
        try:
            doc_source = docs[0].metadata["source"]
        except Exception as e:
            return "Not retrieved"

        if ("pradyunsg" in doc_source.lower()) or ("sphinx" in  doc_source.lower()):
            return "Not retrieved"
        else:
            return doc_source

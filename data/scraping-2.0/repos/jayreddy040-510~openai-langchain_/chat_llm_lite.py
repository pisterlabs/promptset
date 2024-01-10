import os
import uuid
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage, AIMessage
import chromadb

# Load environment variables
load_dotenv()
api_key = os.environ.get("AZURE_OPENAI_API_KEY")
azure_endpoint = os.environ.get("AZURE_OPENAI_API_ENDPOINT")

# Initialize AzureChatOpenAI
azure_chat = AzureChatOpenAI(
    azure_deployment="carecoach-gpt35-16k",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
)

# Create a Chroma client
client = chromadb.EphemeralClient()
collection_name = "chat_collection"
client.create_collection(collection_name)
collection = client.get_collection(collection_name)

# Initialize conversation history
conversation_history = []

# Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break

    # Add new user input to conversation history
    user_message = HumanMessage(content=user_input)
    conversation_history.append(user_message)
    collection.add(ids=[str(uuid.uuid4())], documents=[user_input])

    # Prepare the full conversation context for the LLM query
    full_context = " ".join([f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}" for msg in conversation_history])

    # Get response from AzureChatOpenAI LLM
    llm_response = azure_chat([user_message])
    bot_message = AIMessage(content=llm_response.content if hasattr(llm_response, 'content') else llm_response[0].content)
    print("AI:", bot_message.content)

    # Add the bot's response to the conversation history
    conversation_history.append(bot_message)

from flask import Flask, Blueprint, jsonify, request
from src.datasets.db import supabase
from src.routes.routes import routes
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import SupabaseVectorStore
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import tempfile
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Blueprint and routing details
base = "retrieve"
retrieve_bp = Blueprint(base, __name__)

embeddings = OpenAIEmbeddings()

@retrieve_bp.route(routes[base]["base"], methods=routes[base]["methods"])
def retrieval_agent():
    try:
        # Get the message info from the request
        data = request.get_json()
        channel_id = data.get("channel_id")
        user_message = data.get("message")
        participant_uuid = data.get("participant_id")

        # Get conversation history from database
        message_data_response = supabase.rpc('get_last_n_messages_by_channel_id', {
                                             'channel_uuid': channel_id,
                                             'last_n_messages': 15}).execute()
        print("DATABASE RESPONSE: ", message_data_response)
        message_data_list = message_data_response.data

        # Create a temporary file to store the messages
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            for message in message_data_list:
                f.write(message['content'] + "\n")
            temp_filename = f.name

        #load and parse docs
        loader = TextLoader(temp_filename)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=30, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        # Initialize the vector store
        vector_store = SupabaseVectorStore.from_documents(
            docs,
            embeddings,
            client=supabase,
            table_name="documents",
            query_name="match_documents",
            chunk_size=500,
        )

        # Retrieve the most relevant document
        matched_docs = vector_store.similarity_search(user_message)
        document_content = matched_docs[0].page_content

        # Combine the document content with the user message and generate AI response
        messages = [
            {"role": "system", "content": document_content},
            {"role": "user", "content": user_message}
        ]

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=150
        )

        ai_response = response.choices[0].message.content.strip()

        # Insert the AI response into the database
        supabase.rpc('insert_message', {
            'channel_uuid': channel_id,
            'participant_uuid': participant_uuid,
            'message_content': ai_response,
            'is_agent': True
        }).execute()

        os.remove(temp_filename)

        return jsonify({"message": "agent responded successfully", "response": ai_response})

    except Exception as e:
        print(f"An error occurred: {e}")
        os.remove(temp_filename)
        return jsonify({"error": "Error processing AI response"}), 500




import os
import gradio
import openai
import chromadb
from chromadb.utils import embedding_functions


## Use the following line to connect from within CML
chroma_client = chromadb.PersistentClient(path="/home/cdsw/chroma-data")

EMBEDDING_MODEL_REPO = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
EMBEDDING_FUNCTION = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)

COLLECTION_NAME = os.getenv('COLLECTION_NAME')

print("initialising Chroma DB connection...")
print(f"Getting '{COLLECTION_NAME}' as object...")
try:
    chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)
    print("Success")
    collection = chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)
    # Get latest statistics from index
    current_collection_stats = collection.count()
    print('Total number of embeddings in Chroma DB index is ' + str(current_collection_stats))

except:
    print("Error! Cannot connect to Chroma collection.")


def main():
    # Configure gradio QA app 
    print("Configuring gradio app")
    demo = gradio.Interface(fn=get_responses, 
                            inputs=[gradio.Radio(['gpt-3.5-turbo', 'gpt-4'], label="Select GPT Engine", value="gpt-3.5-turbo"), gradio.Textbox(label="Question", placeholder="")],
                            outputs=[gradio.Textbox(label="Asking Open AI LLM with No Context"),
                                     gradio.Textbox(label="Asking Open AI with Context (RAG)"),
                                     gradio.Textbox(label="File Path(s) Context Reference")],
                            allow_flagging="never")


    # Launch gradio app
    print("Launching gradio app")
    demo.launch(share=True,
                enable_queue=True,
                show_error=True,
                server_name='127.0.0.1',
                server_port=int(os.getenv('CDSW_APP_PORT')))
    print("Gradio app ready")

# Helper function for generating responses for the QA app
def get_responses(engine, question):
    try:
        open_ai_api_key = os.getenv('OPENAI_KEY')
    except:
        return "No OpenAI key declared"
    
    if engine is "" or question is "" or engine is None or question is None:
        return "No question, engine, or api key selected."

    openai.api_key = open_ai_api_key

    
    # Phase 1: Get nearest knowledge base chunk for a user question from a vector db
    results, context_chunk = get_nearest_chunk_from_vectordb(question)

    # Phase 2a: Perform text generation with LLM model using found kb context chunk
    contextResponse = get_llm_response_with_context(question, context_chunk, engine)
    rag_response = contextResponse
    
    # Phase 2b: For comparison, also perform text generation with LLM model without providing context
    plainResponse = get_llm_response_without_context(question, engine)
    plain_response = plainResponse

    return plain_response, rag_response, results

# Get embeddings for a user question and query Chroma vector DB for nearest knowledge base chunk
def get_nearest_chunk_from_vectordb(question):
    
    results = collection.query(
        query_texts=[str(question)],
        n_results=1
        # where={"metadata_field": "is_equal_to_this"}, # sample optional filter
        # where_document={"$contains":"search_string"}  # sample optional filter
    )
    response = ""

    for i in range(len(results['ids'][0])):
        file_path = results['ids'][0][i]

    response += str(load_context_chunk_from_data(file_path))
    
    return str(results), response
  
# Return the Knowledge Base doc based on Knowledge Base ID (relative file path)
def load_context_chunk_from_data(id_path):
    with open(id_path, "r") as f: # Open file in read mode
        return f.read()

  
# Pass through user input to LLM model with enhanced prompt and stop tokens
def get_llm_response_with_context(question, context, engine):
    question = "Answer this question based on given context " + question
    
    response = openai.ChatCompletion.create(
        model=engine, # The deployment name you chose when you deployed the GPT-35-Turbo or GPT-4 model.
        messages=[
            {"role": "system", "content": str(context)},
            {"role": "user", "content": str(question)}
            ]
    )
    return response['choices'][0]['message']['content']


# Pass through user input to LLM model with enhanced prompt and stop tokens
def get_llm_response_without_context(question, engine):
    
    response = openai.ChatCompletion.create(
        model=engine, # The deployment name you chose when you deployed the GPT-35-Turbo or GPT-4 model.
        messages=[
            {"role": "user", "content": str(question)}
            ]
    )
    return response['choices'][0]['message']['content']


if __name__ == "__main__":
    main()

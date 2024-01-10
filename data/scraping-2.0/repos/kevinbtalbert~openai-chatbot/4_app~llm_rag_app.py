import os
import gradio
import openai
from milvus import default_server
from pymilvus import connections, Collection
import utils.vector_db_utils as vector_db
import utils.model_embedding_utils as model_embedding


def main():
    # Configure gradio QA app 
    print("Configuring gradio app")
    demo = gradio.Interface(fn=get_responses, 
                            inputs=[gradio.Radio(['gpt-3.5-turbo', 'gpt-4'], label="Select GPT Engine", value="gpt-3.5-turbo"), gradio.Textbox(label="Question", placeholder="")],
                            outputs=[gradio.Textbox(label="Asking Open AI LLM with No Context"),
                                     gradio.Textbox(label="Asking Open AI with Context (RAG)")],
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

    # Load Milvus Vector DB collection
    vector_db_collection = Collection('cloudera_ml_docs')
    vector_db_collection.load()
    
    # Phase 1: Get nearest knowledge base chunk for a user question from a vector db
    context_chunk = get_nearest_chunk_from_vectordb(vector_db_collection, question)
    vector_db_collection.release()

    # Phase 2a: Perform text generation with LLM model using found kb context chunk
    contextResponse = get_llm_response_with_context(question, context_chunk, engine)
    rag_response = contextResponse
    
    # Phase 2b: For comparison, also perform text generation with LLM model without providing context
    plainResponse = get_llm_response_without_context(question, engine)
    plain_response = plainResponse

    return plain_response, rag_response

# Get embeddings for a user question and query Milvus vector DB for nearest knowledge base chunk
def get_nearest_chunk_from_vectordb(vector_db_collection, question):
    # Generate embedding for user question
    question_embedding =  model_embedding.get_embeddings(question)
    
    # Define search attributes for Milvus vector DB
    vector_db_search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
    
    # Execute search and get nearest vector, outputting the relativefilepath
    nearest_vectors = vector_db_collection.search(
        data=[question_embedding], # The data you are querying on
        anns_field="embedding", # Column in collection to search on
        param=vector_db_search_params,
        limit=2, # limit results to 1
        expr=None, 
        output_fields=['relativefilepath'], # The fields you want to retrieve from the search result.
        consistency_level="Strong"
    )

    # Return text of the nearest knowledgebase chunk
    response = ""
    for f in nearest_vectors[0]:
        response += str(load_context_chunk_from_data(f.id))
    
    return response
  
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

import io
import pinecone
import openai
import functions_framework
from firebase_functions import https_fn
from firebase_functions import options
from firebase_admin import initialize_app

initialize_app()
options.set_global_options(max_instances=10)

@https_fn.on_request(
    cors=options.CorsOptions(
        cors_origins=["http://localhost:3000", "*"],
        cors_methods=["GET", "POST"],
    )
)

@functions_framework.http
def answer_question(request):
    request_json = request.get_json(silent=True)
    request_args = request.args

    # Check if a question was provided in the request
    if "question" not in request_json:
        return "No question provided"


    question = request_json["question"]
    
    index_name = "test"  # Replace with your actual Pinecone index name

    # Initialize Pinecone
    PINECONE_API_KEY = ""
    PINECONE_API_ENV = ""
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

    # Initialize Langchain embeddings
    OPENAI_API_KEY = ""  # Replace with your OpenAI API key
    openai.api_key = OPENAI_API_KEY

    # Convert the user's question into an embedding
    question_embedding = embed_text(question)

    # Search for the most similar embeddings in Pinecone
    index = pinecone.Index(index_name)
    results = index.query(queries=[question_embedding], top_k=3, include_metadata=True)

    # Access the matches correctly
    matches = results['results'][0]['matches']

    relevant_documents = [match['metadata']['text'] for match in matches]

    # Concatenate relevant documents into a single text
    relevant_documents_text = "\n\n".join(relevant_documents)

    if relevant_documents_text == "":
        return "No relevant documents found"
        
    # Create a chat prompt with relevant documents and the user's question
    chat_prompt = f"Relevant Documents:\n{relevant_documents_text}\n\nUser Question: {question}\nAnswer:"
    print(chat_prompt)
    # Generate an answer using GPT-3.5 Turbo
    response = openai.Completion.create(
        engine="text-davinci-002",  # Use the appropriate OpenAI engine
        prompt=chat_prompt,
        max_tokens=50,  # Adjust as needed
    )

    answer = response.choices[0].text
    print(answer)
    return answer



def embed_text(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    embeddings = response['data'][0]['embedding']
    return embeddings

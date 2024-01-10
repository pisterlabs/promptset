import os
from dotenv import load_dotenv
import openai
import pinecone

# Load API keys from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# Initialize OpenAI and Pinecone
openai.api_key = OPENAI_API_KEY
pinecone.init(api_key=PINECONE_API_KEY, environment="eu-west1-gcp")

# Function to embed question using OpenAI
def embed_question(question):
    response = openai.Embedding.create(
        input=question, 
        model="text-embedding-ada-002"
    )
    return response["data"][0]["embedding"]

# Function to query Pinecone for nearest documents
def query_pinecone(question_vector, top_k=10):
    # Initialize Pinecone
    index = pinecone.Index(PINECONE_INDEX)
    

    # Query the index
    query_response = index.query(
        vector=question_vector,
        top_k=top_k,
        include_metadata=True,
        include_values=True,
        namespace=PINECONE_NAMESPACE,
    )

    return query_response



    

def main():
    # Get question from user
    question = input("Enter your question: ")

    # Embed the question using OpenAI
    question_vector = embed_question(question)

    # Query Pinecone for nearest documents
    nearest_documents = query_pinecone(question_vector, 10)

    # Print the results
    with open("output.txt", "w") as file:
        file.write(str(nearest_documents))

    file_ids = [match['id'] for match in nearest_documents['matches']]

    file_contents = []
    for filename in file_ids:
        filepath = f"chunks/{filename}"
        with open(filepath, "r") as file:
            file_contents.append(file.read())

    # Combine the file contents into a single string
    context = "\n".join(file_contents)
    system_message = """
        Answer questions regarding a minecraft mod called terrafirmacraft.
        
        You will be given context that may or may not be related to the question. If the context is related to the question, use it to answer the question. 
        Otherwise, say that you don't know the answer, and attempt to answer it anyways using an 'In General' format.
    """
    prompt = f"""
        Question: {question}

        Context: {context}

        Answer:
    """
    print(f"Prompting {prompt}")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
    )
    print(response['choices'][0]['message']['content'])

if __name__== "__main__":
    main()

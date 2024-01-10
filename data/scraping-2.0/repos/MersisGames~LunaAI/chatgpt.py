import openai
import pandas as pd
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity
import constants
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

# Set up the OpenAI API key
openai.api_key = constants.APIKEY

# Load the PDF and divide it into paragraphs
loader = PyPDFLoader("../whales.pdf")
pages = loader.load_and_split()
split = CharacterTextSplitter(chunk_size=400, separator='. ')
texts = split.split_documents(pages)
texts = [str(i.page_content) for i in texts]  # List of paragraphs
paragraphs = pd.DataFrame(texts, columns=["text"])

# Calculate embeddings
paragraphs['Embedding'] = paragraphs["text"].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))

# Save the paragraphs to a CSV file
paragraphs.to_csv('separated_paragraphs.csv', index=False)

def answer_question(question):
    response = openai.Completion.create(
        engine="text-davinci-003",  # Change the engine to text-davinci-003
        temperature=0,
        max_tokens=100,
        prompt="You can only answer questions about whales. User Input: " + question  # Adjust the number of tokens as needed
    )
    return response.choices[0].text

def search_with_answer(query, data, num_results=5):
    intro_response = answer_question(query)
    
    query_embedding = get_embedding(query, engine="text-embedding-ada-002")
    data["Similarity"] = data['Embedding'].apply(lambda x: cosine_similarity(x, query_embedding))
    data = data.sort_values("Similarity", ascending=False)
    
    detailed_response = data.iloc[0]["text"]
    
    full_response = intro_response + "." + detailed_response
    
    return full_response

text_embeddings = paragraphs

while True:
    question = input("Enter your question ('exit' to quit): ")
    if question.lower() == "exit":
        break
    
    complete_response = search_with_answer(question, text_embeddings)
    print("Response:")
    print(complete_response)

from dotenv import load_dotenv
from openai import OpenAI
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
import json
import google.generativeai as genai


load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
google_api_key = os.environ.get("GOOGLE_AI_API_KEY")

genai.configure(api_key=google_api_key)
client = OpenAI(api_key=api_key)


TOTAL_CHUNK_SIZE = 2000

def chatWithGpt(pdfText : str, userInput : str):

    text_splitter = CharacterTextSplitter(
        separator="",
        chunk_size=TOTAL_CHUNK_SIZE,
        chunk_overlap=0,
        length_function=len,
    )
    
    texts = text_splitter.split_text(pdfText)

    embeddings = OpenAIEmbeddings()

    docsearch = FAISS.from_texts(texts, embeddings)

    docs = docsearch.similarity_search(userInput)
    
    doc_str = ""
    
    for doc in docs:
        doc_str += doc.page_content
    
    system_prompt = f"The following text is provided for users to generate responses.Response type must be JSON and please answer the user's question based on the given document:\n\n{doc_str}\n\nUser's question: {userInput}\n\nAnswer:"

    response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content":system_prompt},
                {"role": "user", "content": userInput}
            ],
            response_format={ "type": "json_object" },    
            frequency_penalty=0.2,
            presence_penalty=0.2,
            temperature=0.7,
    )
    json_str = response.choices[0].message.content
    print(json_str)
    return json.loads(json_str)



def chatWithGoogleAi(pdfText : str, userInput : str):
    text_splitter = CharacterTextSplitter(
        separator="",
        chunk_size=TOTAL_CHUNK_SIZE,
        chunk_overlap=0,
        length_function=len,
    )
    
    texts = text_splitter.split_text(pdfText)

    embeddings = OpenAIEmbeddings()

    docsearch = FAISS.from_texts(texts, embeddings)

    docs = docsearch.similarity_search(userInput)
    
    doc_str = ""
    
    for doc in docs:
        doc_str += doc.page_content
        
    systemPrompt = f"The following text is provided for users to generate responses.Response type must be JSON and please answer the user's question based on the given document:\n\n{pdfText}\n\nUser's question: {userInput}\n\nAnswer:"
    
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(
        systemPrompt)
    return response.text

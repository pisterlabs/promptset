# import libraries
import pandas as pd
import io
import spacy
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.agents import create_pandas_dataframe_agent

# get multiple chunks from the text
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


# get vector
def get_vectorstore(text_chunks):
    # generate embddings
    embeddings =  OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# handle user question
def  handleQuestion(question, vectorstore):
    chat_history = []
    model = ChatOpenAI(model_name="gpt-3.5-turbo")  # switch to 'gpt-4'            
    qa = ConversationalRetrievalChain.from_llm(model, retriever=vectorstore.as_retriever())
    result = qa({"question": question, "chat_history": chat_history})
    # Remove the 'chat_history' key from the result
    result.pop("chat_history", None)
    return result

# // function to process csv
def processCSV(filename, question):
    llm = ChatOpenAI(temperature=0)
    agent = create_pandas_dataframe_agent(llm, filename)
    response = agent.run(question)
    answer = {}
    answer['question'] = question
    answer['answer'] = response
    return answer

# // decode csv files
def decode_csv(decoded_data):
    # Decode CSV data and return it in a structured format
    df = pd.read_csv(io.BytesIO(decoded_data))
    # You can further process the data as needed, or return the DataFrame
    return df

# // calculate number of words, sentences and characters
def countWords(data):
    # Calculate the number of words
    num_words = len(data.split())

    # Calculate the number of characters
    num_chars = len(data)

    # Calculate the number of sentences
    num_sentences = len(data.split("."))

    nlp = spacy.load("en_core_web_sm")
    # Tokenize the text and identify the parts of speech for each token.
    doc = nlp(data)
    pos_tags = [token.pos_ for token in doc]
    
    # Count the number of tokens for each part of speech.
    pos_tag_counts = {}
    for pos_tag in pos_tags:
        if pos_tag not in pos_tag_counts:
            pos_tag_counts[pos_tag] = 0
        pos_tag_counts[pos_tag] += 1
    return num_chars, num_words, num_sentences, pos_tag_counts,
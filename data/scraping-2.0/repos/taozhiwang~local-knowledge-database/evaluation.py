from langchain.chains import RetrievalQA
from langchain.evaluation.loading import load_dataset
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
import openai
from langchain.llms import OpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os

def evaluation(embedding_type, need_embedding):
    dataset = load_dataset("question-answering-paul-graham")
    loader = TextLoader("./data/paul_graham_essay.txt")
    data = loader.load()
    # Create split file
    text_splitter = CharacterTextSplitter(chunk_size=40, chunk_overlap=5)
    split_docs = text_splitter.split_documents(data)
    docs = split_docs
    if need_embedding:
        if embedding_type == "HuggingFace":
            embeddings = HuggingFaceEmbeddings()
        else:
            assert(False)
        #document_store = FAISSDocumentStore(sql_url = 'embedding.db')
        #document_store.write_documents(datadict)

    db = FAISS.from_documents(split_docs, embeddings)

    query = dataset[0]["question"]
    print(query)
    return
    similar_docs = db.similarity_search(query)

    # Create prompt
    system_prompt = "You are a person who answers questions for people based on specified information\n"
    similar_prompt = similar_docs[0].page_content + "\n" + similar_docs[1].page_content + "\n" + similar_docs[2].page_content + "\n"
    question_prompt = f"Here is the question: {query}\nPlease provide an answer only related to the question and do not include any information more than that.\n"
    prompt = system_prompt + "Here is some information given to you:\n" + similar_prompt + question_prompt

    # Openai api setting
    openai.api_key = os.environ["OPENAI_API_KEY"]

    print(prompt)
    # Create response from gpt-3.5
    response = openai.ChatCompletion.create(
    model = "gpt-3.5-turbo",
    temperature =  0.7,
    messages=[
            {"role": "system", "content": "你是一个根据提供的材料作出回答的助手"},
            {"role": "user", "content": prompt},
        ],
    max_tokens = 40
    )

    print(response['choices'][0]['message']['content'].replace(' .', '.').strip())


if __name__ == "__main__":
    evaluation("HuggingFace", True)
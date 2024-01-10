from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings

embeddings_model_name = "hkunlp/instructor-xl"
embeddings = HuggingFaceInstructEmbeddings(model_name=embeddings_model_name, model_kwargs={"device": "cuda"})

with open("Functions/data.txt", "r") as text:
    text = text.read()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len
        )
    chunks = text_splitter.split_text(text=text)

    # Embed chunks and store them in a Vector Store
    VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

query = 'what is a nebula'
if query:
    docs = VectorStore.similarity_search(query=query, k=3)

    llm = GooglePalm()
    llm.temperature = 0.1
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    response = chain.run(input_documents=docs, question=query)

print(response)
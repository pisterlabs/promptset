def run_rag_pipeline(query, model="gpt-3.5-turbo-instruct", dataset="nfcorpus"):
    # Imports
    from dotenv import load_dotenv
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.llms import OpenAI
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema.output_parser import StrOutputParser
    from langchain.schema.runnable import RunnablePassthrough
    from langchain.vectorstores import FAISS

    import os
    import pickle

    # Load index from file
    load_dotenv()

    loaded_faiss_vs = FAISS.load_local(
        folder_path=f"./vectordb/faiss/{dataset}/",
        embeddings=OpenAIEmbeddings())
    retriever = loaded_faiss_vs.as_retriever(search_kwargs={"k": 5})

    # Define the RAG pipeline
    llm = OpenAI(model_name=model, openai_api_key=os.getenv("OPENAI_API_KEY"))

    template = """Answer the question or Explain the topic given this additional context: {context}
    Question: {question}"""
    prompt = ChatPromptTemplate.from_template(template)

    docs_file_path = f"./dataset/{dataset}/documents.pkl"
    with open(docs_file_path, "rb") as file:
        docs = pickle.load(file)

    def format_docs(_docs):
        ls = []
        for doc in _docs:
            if doc.page_content in docs:
                ls.append(docs[doc.page_content]["text"][:800])
        return ls

    chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()}
             | prompt
             | llm
             | StrOutputParser())

    # Run the RAG pipeline
    response = chain.invoke(query)
    return response


if __name__ == "__main__":
    print(run_rag_pipeline(
        query="Do Cholesterol Statin Drugs Cause Breast Cancer?",
        dataset="nfcorpus"))
    print(run_rag_pipeline(
        query="Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?",
        dataset="pubmedqa"))
    print(run_rag_pipeline(
        query="what is the origin of COVID-19 (seeking range of information about the SARS-CoV-2 virus's origin, "
              "including its evolution, animal source, and first transmission into humans)",
        dataset="cord19"))

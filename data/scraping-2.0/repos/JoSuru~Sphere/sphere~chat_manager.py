from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma

from sphere.utils import embedding_model, llm_loader


def main(query):
    persist_directory = "db"

    # Now we can load the persisted database from disk, and use it as normal.
    vectordb = Chroma(
        persist_directory=persist_directory, embedding_function=embedding_model()
    )
    # Build prompt
    template = """Utilisez les éléments de contexte suivants pour répondre à la question finale. Si vous ne 
    connaissez pas la réponse, dites simplement que vous ne savez pas, n'essayez pas d'inventer une réponse.
    La réponse doit être aussi aussi concise que possible. Dites toujours "merci d'avoir posé la question" à
    la fin de la réponse. Tu dois toujours répondre à la question en français et dans le format markdown.
    Réfléchi étape par étape.
    {context}
    Question: {question} 
    Réponse utile :"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)  # Run chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_loader(),
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )
    result = qa_chain({"query": query})
    return result["result"]

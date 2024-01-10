from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from utils import load_embeddings, load_db


load_dotenv()

class NavItzamnaChat:

    def __init__(self, idx_name) -> None:
        
        embedding_function = load_embeddings()
        db = load_db(embedding_function, index_name = idx_name)
        llm = ChatOpenAI(temperature=0.1)
        self.qa = RetrievalQA.from_llm(llm=llm, retriever=db.as_retriever(kwargs={"k": 7}), return_source_documents=True)

    def answer_question(self, question :str):
        output = self.qa({"query": question})
        source_docs =' '.join(["\n"+str(doc.metadata) for doc in output["source_documents"]])
        print("\nDocumentos de los que se obtuvo la respuesta:" + source_docs + "\n")
        return output["result"]



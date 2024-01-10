from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.chains import ConversationalRetrievalChain

class RepoAgent:
  def __init__(self, deeplake_username="", deeplake_db="", model="gpt-3.5-turbo-16k"):
    self.deeplake_username = deeplake_username
    self.deeplake_db = deeplake_db
    self.model = model
    self.chat_history = []
    self.embeddings = OpenAIEmbeddings(disallowed_special=())
    self.db = DeepLake(
      dataset_path=f"hub://{self.deeplake_username}/{self.deeplake_db}",
      read_only=True,
      embedding_function=self.embeddings,
    )
    self.init_qa()
  
  def init_qa(self):
    retriever = self.db.as_retriever()
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["fetch_k"] = 100
    retriever.search_kwargs["maximal_marginal_relevance"] = True
    retriever.search_kwargs["k"] = 10
    model = ChatOpenAI(model_name=self.model)
    self.qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)
  
  def ask(self, question):
    result = self.qa({"question": question, "chat_history": self.chat_history})
    self.chat_history.append((question, result["answer"]))
    print(f"-> **Question**: {question} \n")
    print(f"**Answer**: {result['answer']} \n")
    return result['answer']
from langchain.vectorstores import Chroma


class VECDB:
    def __init__(self, doc, em):
        self.persist_directory='./vecbd'
        self.vectordb = Chroma.from_documents(
            documents=doc,  # 为了速度，只选择了前 100 个切分的 doc 进行生成。
            embedding=em,
            persist_directory=self.persist_directory  # 允许我们将persist_directory目录保存到磁盘上
        )
        self.vectordb.persist()

    def search(self, question):
        self.sim_docs = self.vectordb.similarity_search(question,k=3)
        return self.sim_docs


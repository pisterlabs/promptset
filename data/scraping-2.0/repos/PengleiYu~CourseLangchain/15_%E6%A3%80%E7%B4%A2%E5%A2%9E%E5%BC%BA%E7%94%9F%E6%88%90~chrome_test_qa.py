# 持久化，读取pdf并查询
from chromadb.config import Settings
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores.chroma import Chroma

settings = Settings(is_persistent=True)

loader = PyPDFLoader(file_path='../02_文档QA系统/docs/易速鲜花员工手册.pdf', )

creator = VectorstoreIndexCreator(vectorstore_cls=Chroma, vectorstore_kwargs={
    'client_settings': settings,
})
vector_store_index_wrapper = creator.from_loaders(loaders=[loader])

question_list = [
    # '董事长是谁？',
    '易速鲜花工资哪天发？',
]
for question in question_list:
    print(question)
    result = vector_store_index_wrapper.query(question)
    print('     ', result)

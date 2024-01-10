from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from utils.AliTextSplitter import AliTextSplitter

from config import Config
from document import DocumentService
from chatGLM import LLMService
from custom_search import DeepSearch

def get_related_content(related_docs):
    related_content = []
    for doc in related_docs:
        related_content.append(doc.page_content)
    return "\n".join(related_content)

def get_docs_with_score(docs_with_score):
    docs = []
    for doc, score in docs_with_score:
        doc.metadata["score"] = score
        docs.append(doc)
    return docs


class LangChainApplication(object):

    def __init__(self):
        self.config = Config
        self.llm_service = LLMService()
        ###加载llm和知识库向量
        print("load llm model ")
        self.llm_service.load_model(model_name_or_path=self.config.llm_model_name)
        self.doc_service = DocumentService()
        print("load documents")
        self.doc_service.load_vector_store()

    def get_knowledge_based_answer(self, query,
                                   history_len=5,
                                   temperature=0.1,
                                   top_p=0.9,
                                   top_k=1,
                                   chat_history=[]):
        #定义prompt
        prompt_template = """基于以下已知信息，简洁和专业的来回答用户的问题。
                                        如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。
                                        已知内容:
                                        {context}
                                        问题:
                                        {question}"""
        prompt = PromptTemplate(template=prompt_template,
                                input_variables=["context", "question"])
        self.llm_service.history = chat_history[-history_len:] if history_len > 0 else []

        self.llm_service.temperature = temperature
        self.llm_service.top_p = top_p


        related_docs_with_score = self.doc_service.vector_store.similarity_search_with_score(query, k = top_k)
        related_docs = get_docs_with_score(related_docs_with_score)
        if related_docs[0].metadata["score"] > 500:


            # 声明一个知识库问答llm,传入之前初始化好的llm和向量知识搜索服务
            knowledge_chain = RetrievalQA.from_llm(
                llm=self.llm_service,
                retriever=self.doc_service.vector_store.as_retriever(
                    search_kwargs={"k": top_k}),
                prompt=prompt)

        else:
            result_str = DeepSearch.search(query)
            text_splitter = AliTextSplitter()
            # 使用阿里的分段模型对文本进行分段
            split_text = text_splitter.split_text(result_str)
            # 采用embeding模型对文本进行向量化
            
            vector_store = FAISS.from_texts(split_text, self.doc_service.embeddings)
            vector_store.save_local("resource")

            knowledge_chain = RetrievalQA.from_llm(
                llm=self.llm_service,
                retriever=vector_store.as_retriever(
                    search_kwargs={"k": top_k}),
                prompt=prompt)


        knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
            input_variables=["page_content"], template="{page_content}")
        knowledge_chain.return_source_documents = True

        ### 基于知识库的问答
        result = knowledge_chain({"query": query})


        return result

    def get_llm_answer(self, query=''):
        prompt_template = """请回答下列问题:
                            {}""".format(query)
        ### 基于大模型的问答
        result = self.llm_service._call(prompt_template)
        return result


if __name__ == '__main__':
    application = LangChainApplication()

#     print("大模型自己回答的结果")
#     result = application.get_llm_answer('北京')
#     print(result)
#     print("大模型+知识库后回答的结果")
#     result = application.get_knowledge_based_answer('北京')
#     print(result)
    
    print("大模型自己回答的结果")
    result = application.get_llm_answer('2023年6月13号上海天气怎么样')
    print(result)
    print("大模型+知识库后回答的结果")
    result = application.get_knowledge_based_answer('2023年6月13号上海天气怎么样')
    print(result)

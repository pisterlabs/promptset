import os
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.memory import ConversationBufferMemory
from fastapi import FastAPI
from langserve import add_routes



# (회사) 유료 API 키!!!!!!!!
# 20230904_AIR	
os.environ['OPENAI_API_KEY'] = "YOUR OPENAI API KEY !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

template = """You are a cyber security analyst. about user question, answering specifically in korean.
            Use the following pieces of context to answer the question at the end. 
            You mast answer after understanding previous conversation.
            If you don't know the answer, just say that you don't know, don't try to make up an answer. 
            For questions, related to Mitre Att&ck, in the case of the relationship between Tactics ID and T-ID (Techniques ID), please find T-ID (Techniques ID) based on Tactics ID.
            Tactics ID's like start 'TA' before 4 number.
            T-ID (Techniques ID) like start 'T' before 4 number.
            Tactics ID is a major category of T-ID (Techniques ID), and has an n to n relationship.
            Respond don't know to questions not related to cyber security.
            Use three sentences maximum and keep the answer as concise as possible. 
            context for latest answer: {context}
            Previous conversation: 
            {chat_history}
            latest question: {question}
            latest answer: """
            
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "chat_history", "question"],template=template)

db_save_path = "YOUR DB SAVE PATH !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

embeddings = OpenAIEmbeddings()

# callbacks = [StreamingStdOutCallbackHandler()]
# chat_llm = OpenAI(model_name='gpt-3.5-turbo-instruct', temperature=0, max_tokens=512,
#                   # callbacks=callbacks, 
#                   # doc 이 100개가 넘는 경우 OpenAI API 호출이 안되므로 false 처리 !!!!!!!!!!!!!
#                   # streaming=True
#                   )
chat_llm = ChatOpenAI(model_name='gpt-4-1106-preview', temperature=0, max_tokens=512,
                  # callbacks=callbacks, 
                  # doc 이 100개가 넘는 경우 OpenAI API 호출이 안되므로 false 처리 !!!!!!!!!!!!!
                  # streaming=True
                  )

new_docsearch = FAISS.load_local(os.path.join(db_save_path, 'mitre_attack_20231129_index'), embeddings)

retriever = new_docsearch.as_retriever(search_type="similarity", search_kwargs={"k":2,
                                                                                "score_threshold": 0.7}
                                                                                )
# 유사도 0.7 이상만 추출
# embeddings_filter = EmbeddingsFilter(embeddings = embeddings, similarity_threshold = 0.7)

# # 압축 검색기 생성
# compression_retriever = ContextualCompressionRetriever(base_compressor = embeddings_filter,
#                                                         base_retriever = retriever)


memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, input_key = "question", output_key='answer')
retrieval_qa_chain = ConversationalRetrievalChain.from_llm(llm = chat_llm,
                                        # retriever = compression_retriever,
                                        retriever = retriever,

                                        memory = memory,
                                        return_source_documents = True,
                                        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
                                        )

class SimpleJsonOutputParser:
    def __call__(self, result):
        return result["answer"]

retrieval_qa_chain_pipe = retrieval_qa_chain | SimpleJsonOutputParser()

app = FastAPI(title="Mitre Att&ck App",
                version="1.0",
                description="A simple API server using LangChain's Runnable interfaces")
# Add the LangServe routes to the FastAPI app

# 3. Adding chain route
add_routes(
    app,
    # retrieval_qa_chain,
    retrieval_qa_chain_pipe,

    path="/retrieval_qa_chain",
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
    

'''
/docs: langserve API 문서

Playground UI
http://localhost:8000/retrieval_qa_chain/playground/
'''

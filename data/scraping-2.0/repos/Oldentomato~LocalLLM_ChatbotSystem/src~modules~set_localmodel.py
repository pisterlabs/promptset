from langchain.vectorstores import Chroma
import os
from transformers import AutoModelForCausalLM, LlamaTokenizerFast, pipeline
from langchain import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from .custom.textstreamer import TextStreamer
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from .embedding import Embedding_Document
from langchain.schema.retriever import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from typing import List

#"beomi/llama-2-ko-7b"
#"jhgan/ko-sroberta-multitask"

class URRetrival(BaseRetriever):
    doc_embedding:Embedding_Document
    mode:str
    k:int

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        if self.mode == "bmbert":
            content, source, page, score = self.doc_embedding.search_doc_bm_bert(query, k=1, bm_k=self.k)
            content = list(map(lambda x:Document(page_content=x, metadata={"source":source,"page":page}), content))
        # response = URAPI(request)
        # convert response (json or xml) in to langchain Document like  doc = Document(page_content="response docs")
        # dump all those result in array of docs and return below
        return content


class Set_LocalModel:
    def __init__(self):
        self.model = "beomi/llama-2-ko-7b"
        self.chat_history = []
        self.context = ""
        self.doc_embedd = Embedding_Document(
            save_tfvector_dir = "/prj/src/tf_data_store",
            save_doc2vec_dir = "/prj/src/doc2vec_data_store",
            save_bert_dir = "/prj/src/bert_data_store",
            save_bm25_dir = "/prj/src/bm25_data_store"
        )


    def read_summary(self):
        with open("/prj/src/data_store/summary.txt", "r") as file:
            self.summary = file.read()

        print(self.summary)


    def get_llm_model(self):
        print("model load")
        self.pre_model = AutoModelForCausalLM.from_pretrained(
        self.model,
        device_map="auto",
        load_in_8bit=True,
        trust_remote_code=True
        )

        self.pre_model.eval()
        self.tokenizer = LlamaTokenizerFast.from_pretrained(self.model)



        # 대화 내역을 활용하여 추가적으로 답변을 생성해도 됩니다. 대화 내역은 아래와 같습니다.\n
        # {chat_history}
    def __set_prompt(self):
        prompt_template = """당신은 문서검색 지원 에이전트입니다.\n
        전반적인 내용은 아래와 같습니다. 이 내용을 이용해서 답변을 해주세요.\n
        {context}
        답을 모르면 모른다고만 하고 답을 만들려고 하지 마세요. 같은 말은 반복하지 마세요.\n
        참고된 pdf의 내용이 없다면, 없다고 답하세요.\n
        """
        system_prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context"]
        )

        system_message_prompt = SystemMessagePromptTemplate(prompt=system_prompt)

        question_prompt = PromptTemplate(
            template="""아래 문장에 대한 답변을 말해주세요.\n
            {question}\n
            답변:""",
            input_variables=["question"]
        )

        question_message_prompt = HumanMessagePromptTemplate(prompt=question_prompt)

        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, question_message_prompt])


        return chat_prompt
    

    def pdf_embedding(self, pdfs, embedding_mode="tf-idf"):
        if embedding_mode == "bert":
            success, e = self.doc_embedd.embedding_bert(pdfs)
        elif embedding_mode == "tf-idf":
            success, e = self.doc_embedd.embedding_tf_idf(pdfs)

        elif embedding_mode == "doc2vec":
            success, e = self.doc_embedd.embedding_doc2vec(pdfs)

        elif embedding_mode == "bm25":
            success, e = self.doc_embedd.bm25_embedding(pdfs)


        return success, e


    def search_doc(self, query, k, embedding_mode="bert"):
        if embedding_mode == "tf-idf":
            content, source, page, score = self.doc_embedd.tf_idf_search_doc(query, k)
        elif embedding_mode == "doc2vec":
            content, source, page, score = self.doc_embedd.doc2vec_search_doc(query, k)
        elif embedding_mode == "bert":
            content, source, page, score = self.doc_embedd.bert_search_doc(query, k)
        elif embedding_mode == "bm25":
            content, source, page, score = self.doc_embedd.bm25_search_doc(query, k)
        elif embedding_mode == "bmbert":
            content, source, page, score = self.doc_embedd.search_doc_bm_bert(query, bm_k=k, k=1)


        return content, source, page, score

    

    def run_QA(self, g, question, embedding_mode="bmbert"):
        try:
            # db = Chroma(persist_directory="/prj/src/data_store" , embedding_function=self.embeddings)
            # db.get() 

            streamer = TextStreamer(g=g, tokenizer=self.tokenizer, skip_prompt=True)

            pipe = pipeline(
                "text-generation", 
                model=self.pre_model, 
                repetition_penalty=1.1, #높을수록 반복성을 방지시킴
                tokenizer=self.tokenizer, 
                return_full_text = False, 
                max_new_tokens=200, #max_length는 프롬프트를 포함한 최대길이를 지정, max_new_tokens는 프롬프트를 제외한 최대길이를 지정함
                streamer=streamer,
            )
            # res = pipe( #__call__
            #     do_sample = True, # False로 해두면 Greedy Search만 진행되어 같은 결과만 생성하게 된다.
            #     no_repeat_ngram_size=2, #생성시간이 숫자에 비례해서 올라가지만 문장 생성시에 반복되는 경우를 줄일 수 있다.
            #     temperature=0.2, # 낮을수록 가장 확률이 높은 토큰을 선택하게됨. 높을수록 더 창의적인 답변이 나오게됨 (낮음: 사실적, 높음: 창의적)
            #     #top_k와 top_p는 greedy가 False(do_sample이 True)인 경우에 활용된다.
            #     top_k=0, #확률이 높은 순서대로 k번째까지 높은 단어에 대해 필터링한다. 값이 높을수록 무작위 샘플 방식에 가까워지고 낮을수록 탐욕 방식에 가까워진다.
            #     top_p=0.95, #여기서 top_k를 0으로두고 top_p의 값만 할당해주면 뉴클러스 샘플링이라고 하게 된다. 확률이 가장 높은 순으로 후보 단어들을 더했을 때 top_p가 되는단어 집합에 대해 샘플링한다.
                
            #     #즉, greedy를 True로 하면 항상 확률이 높은 단어만 선택하기때문에 편향적이고 일관된 문장만 출력하게되고, 경우에 따라 반복되는 단어가 나올수 있고,
            #     #greedy를 False로 하면 top_k와top_p의 샘플링 방식을 통해 좀 더 자연스럽고 다양한 문장들을 출력하게 된다.


            # )
            hf_model = HuggingFacePipeline(pipeline=pipe,
                                            model_kwargs={
                                                "do_sample": True,
                                                "no_repeat_ngram_size": 2,
                                                "temperature": 0.2,
                                                "top_k": 0,
                                                "top_p": 0.95
                                            })

            memory = ConversationBufferMemory(memory_key="chat_history", human_prefix="question", ai_prefix="answer", return_messages=True)

            # Set retriever and LLM
            # retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3}) # 3개의 답변을 생성하고 유사도를 기준으로 가장 높은 점수를 최종답변으로 도출함
            retriever = URRetrival(doc_embedding=self.doc_embedd, mode=embedding_mode, k=3)


            #compression_retriever
            print("qa_chain load")
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=hf_model,
                chain_type="stuff", #map_rerank
                retriever=retriever,
                return_source_documents=True,
                memory=memory,
                combine_docs_chain_kwargs={"prompt":self.__set_prompt()}
                )


            #, "context" : self.context, "chat_history": self.chat_history
            response = qa_chain({"question":question})
            # self.chat_history= [(question, response["answer"])]

            g.send(f"\n파일: {os.path.basename(response['source_documents'][0].metadata['source'])}") #3개중에 하나고르는거다보니까 0번으로해버리면 정확해지지가 않음
            g.send(f"\n페이지: {response['source_documents'][0].metadata['page']}")


        except Exception as e:
            print('Failed:'+str(e))
        finally:
            g.close()


    
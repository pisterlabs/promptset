
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chat_models import ChatOpenAI
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import GraphQAChain
from langchain.indexes.graph import NetworkxEntityGraph
import re
from langchain.callbacks import get_openai_callback
from utils.count_tokens import log_token_details_to_file
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
import json
import os
import re
from utils.custom_chain import CustomConversationalRetrievalChain

def remove_sources(text):
    # pattern = re.compile(r'\b(SOURCE[S]?:(.*$))', re.IGNORECASE | re.DOTALL)
    
    # cleaned_text = pattern.sub('', text)
    cleaned_text = "\n".join(line.split(" (Source:")[0] for line in text.split("\n"))
    pattern = re.compile(r'\b(SOURCE[S]?:(.*$))', re.IGNORECASE | re.DOTALL)
    cleaned_text = pattern.sub('', cleaned_text)
    return cleaned_text.strip()


def get_citations(response):
    citations = []
    idx = 1
    for d in response["input_documents"]:
        cited_text = "<b>" + f"[{idx}] File Name of Source: " + d.metadata["source"] + "</b>" + "<br>" + d.page_content
        citations.append (cited_text)
        idx+=1
    return citations

def get_citations_v2(response):
    citations = []
    idx = 1
    for d in response["source_documents"]:
        try:
            current_directory = os.path.dirname(os.path.abspath(__file__))
            url_mapping_path = os.path.join(current_directory, '..', 'data', 'title_url_mapping.json')
            with open (url_mapping_path) as f:
                title_url_mapping = json.load(f)
        except:
            print ("No title_url_mapping.json found")
            pass
        
        source_text = d.metadata["source"]
        IF_SOURCE_IS_URL = False
        try:
            dict_key = d.metadata["source"].split("./")[1]
            source_text = title_url_mapping[dict_key]
            IF_SOURCE_IS_URL = True
            print ("dict_key: ", dict_key)
        except:
            print ("No dict_key found")
            pass
        
        cited_text = "<b>"  + source_text + "</b>" + "<br>" + d.page_content
        if IF_SOURCE_IS_URL:
            cited_text = "<b><a href='" + source_text + "'>" + source_text + "</a></b><br>" + d.page_content;
        citations.append (cited_text)
        idx+=1
    return citations

def sorted_doc(all_docs, all_scores):
    paired_results = list(zip(all_docs, all_scores))
    sorted_pairs = sorted(paired_results, key=lambda x: x[1], reverse=True)
    topK_docs = [pair[0] for pair in sorted_pairs]
    return topK_docs

class QueryDocs():
    def __init__(self,
                pinecone_api_key=None,
                pinecone_env_name=None,
                pinecone_index_name=None,
                model_version="gpt-3.5-turbo-16k"):
        
        self.pinecone_api_key=pinecone_api_key
        self.pinecone_env_name=pinecone_env_name
        self.pinecone_index_name=pinecone_index_name
        self.embeddings = OpenAIEmbeddings()
        self.model_version = model_version
        print ("use model version: ", self.model_version)
        pinecone.init(api_key=self.pinecone_api_key,environment=self.pinecone_env_name)
        self.index = pinecone.Index(self.pinecone_index_name)
    
    def qa_pdf (self, question, my_namespace="unilever", text_key="text", topK=5):
        vectorstore = Pinecone(self.index , self.embeddings.embed_query, text_key, namespace=my_namespace)
        docs = vectorstore.similarity_search(question, k=topK)
        chain = load_qa_with_sources_chain(ChatOpenAI(model=self.model_version ,temperature=0), chain_type="stuff")
        response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
        return response["output_text"]

    def qa_knowledge_triples(self, question, graph_pth="ps-graph.gml"):
        loaded_graph = NetworkxEntityGraph.from_gml(graph_pth)
        chain = GraphQAChain.from_llm(ChatOpenAI(model=self.model_version , temperature=0), graph=loaded_graph, verbose=True)
        response = chain.run(question)
        return response
    
    def qa_pdf_with_citations (self, question, my_namespace="unilever", text_key="text", topK=5):
        vectorstore = Pinecone(self.index , self.embeddings.embed_query, text_key, namespace=my_namespace)
        docs = vectorstore.similarity_search(question, k=topK)
        chain = load_qa_with_sources_chain(ChatOpenAI(model=self.model_version ,temperature=0), chain_type="stuff")
        question = question + "Try to summarise your answer in a list. Make sure each item in a list is detailed, but does not have overlap content."
        response = chain({"input_documents": docs, "question": question}, return_only_outputs=False)
        response["citations"] = get_citations(response)
        print (response)
        return response
    
    def qa_pdf_with_citations_from_multiple_srcs (self, question, namespaces_list=["unilever"], text_key="text", topK=5):

        all_docs = []
        for namespace in namespaces_list:
            vectorstore = Pinecone(self.index, self.embeddings.embed_query, text_key, namespace=namespace)
            docs = vectorstore.similarity_search(question, k=topK)
            for doc in docs:
                if len(doc.page_content) > 200: # filter out short documents
                    all_docs.append(doc)

        chain = load_qa_with_sources_chain(ChatOpenAI(model=self.model_version,temperature=0), chain_type="stuff")
        print ("model version: ", self.model_version)
        with get_openai_callback() as cb:
            question = question + "Try to summarise your answer in a list. Make sure each item in a list is in as many details as possible, but does not have overlap content."
            response = chain({"input_documents": all_docs, "question": question}, return_only_outputs=False)
            response["citations"] = get_citations(response)
            response["output_text"] = re.sub(r'^SOURCES:.*$', '', response["output_text"], flags=re.MULTILINE)
            
            # log token details
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            log_token_details_to_file(cb.prompt_tokens, cb.completion_tokens, self.model_version)

        return response

    def qa_pdf_with_conversational_chain (self, question, chat_history, my_namespace=["unilever"], text_key="text", topK=10):
        print ("my namespace: ", my_namespace)

        vectorstore = Pinecone(self.index , self.embeddings.embed_query, text_key)
        llm = ChatOpenAI(model=self.model_version,temperature=0)
        qllm = ChatOpenAI(model='gpt-4',temperature=0)
        doc_chain = load_qa_with_sources_chain(llm, chain_type="stuff")
        
        question_generator = LLMChain(llm=qllm, prompt=CONDENSE_QUESTION_PROMPT)
        
        all_docs = []
        all_scores = []
        for namespace in my_namespace:
            vectorstore = Pinecone(self.index, self.embeddings.embed_query, text_key, namespace=namespace)
            docs = vectorstore.similarity_search(question, k=topK)
            search_result = vectorstore.similarity_search_with_score (question, k=topK)
            for s in search_result:
                doc = s[0]
                score = s[1]
                if score >= 0.85:
                    all_docs.append(doc)
                    all_scores.append(score)
        ref_docs = sorted_doc(all_docs, all_scores)

        question = question + 'Requiremnt of answer: Please group your final answer in a list format if necessary and explain each item in as many details as possible, but does not have overlap content.'
        if len(ref_docs) == 0:
            response = {}
            response["output_text"] = """Apologies, I was unable to retrieve relevant information from the documents. For more accurate results, please consider asking a more specific question. For instance, instead of inquiring, 'What does the 7th item in the list mean?', you might ask, 'What does risk response mean in your previous answer?'.
                                      """
            response["citations"] = []
            return response
        
        chain = CustomConversationalRetrievalChain(
                    retriever=vectorstore.as_retriever(),
                    question_generator=question_generator,
                    combine_docs_chain=doc_chain,
                    return_source_documents=True,
                )
        
        chain.set_input_docs(ref_docs)
        
        with get_openai_callback() as cb:
            print ("input chat_history: ", chat_history)
            response = chain({"question": question, "chat_history": chat_history})
            response["output_text"] = response["answer"]
            try:
                response["output_text"] = remove_sources(response["output_text"])
            except:
                pass
            print ("response: ", response["output_text"])
            if len(ref_docs)>0:
                response["citations"] = get_citations_v2(response)
            
            # log token details
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            log_token_details_to_file(cb.prompt_tokens, cb.completion_tokens, self.model_version)
        return response        
    
    def qa_pdf_with_conversational_chainV2(self, question, chat_history, my_namespace="my-pdf", text_key="text"):

        print ("my namespace: ", my_namespace)
        question = question + ' Requiremnt: Please group your final answer in a list format if necessary and explain each item in as many details as possible, but does not have overlap content.'
        print ("question: ", question)
        vectorstore = Pinecone(self.index, self.embeddings, text_key, namespace=my_namespace)

        llm = ChatOpenAI(model=self.model_version, temperature=0)
        qllm = ChatOpenAI(model=self.model_version, temperature=0)

        doc_chain = load_qa_with_sources_chain(llm, chain_type="stuff")
        
        question_generator = LLMChain(llm=qllm, prompt=CONDENSE_QUESTION_PROMPT)

        chain = ConversationalRetrievalChain(
                    # retriever=vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.8}),
                    retriever=vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'fetch_k': 30}),
                    question_generator=question_generator,
                    combine_docs_chain=doc_chain,
                    return_source_documents=True,
                )
                
        with get_openai_callback() as cb:
            print ("input chat_history: ", chat_history)
            response = chain({"question": question, "chat_history": chat_history})
            response["output_text"] = response["answer"]
            print ("response: ", response["output_text"])
            try:
                response["output_text"] = remove_sources(response["output_text"])
            except:
                pass
            print ("response: ", response["output_text"])
            try:
                response["citations"] = get_citations_v2(response)
            except:
                pass

            # log token details
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            log_token_details_to_file(cb.prompt_tokens, cb.completion_tokens, self.model_version)
        return response   
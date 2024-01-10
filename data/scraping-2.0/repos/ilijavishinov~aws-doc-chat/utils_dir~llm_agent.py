import logging

from langchain.chat_models import ChatOpenAI
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, AutoModel, RobertaForCausalLM, AutoModelForQuestionAnswering
from langchain import HuggingFacePipeline
from langchain.llms import HuggingFacePipeline
from langchain.llms import LlamaCpp, GPT4All
from langchain.chains import RetrievalQA
import os
from utils_dir.text_processing import console_print
from langchain.prompts import PromptTemplate


class LlmAgent(object):
    llm = None
    
    def __init__(self,
                 llm_model_name: str = None):
        self.llm_model_name = llm_model_name
        self.rag_prompt_template = """Use only the following pieces of context to answer the question at the end. \
        If the context does not contain the answer, say that the documentation does not contain the answer.

        {context}

        Question: {question}
        Answer:"""
        self.llm_rag_prompt = PromptTemplate(
            template = self.rag_prompt_template, input_variables = ["context", "question"]
        )
        self.get_llm_object()
    
    def get_llm_object(self):
        """
        Returns the LLM object based on the provided model name
        """
        
        if self.llm_model_name.startswith('openai'):
            self.llm = ChatOpenAI(model_name = "gpt-3.5-turbo")
            
        elif self.llm_model_name == 'llamacpp':
            self.llm = LlamaCpp(
                model_path = r'C:\Users\ilija\llama.cpp\models\7B\ggml-model-q4_0.gguf',
                verbose = True,
                n_ctx = 1024,
                n_threads = 8,
                n_gpu_layers = 40,
                n_batch = 512)
            
        elif self.llm_model_name == 'gpt4all':
            self.llm = GPT4All(
                model = './models/ggml-gpt4all-j-v1.3-groovy.bin',
            )
            # verbose = True, n_ctx = 1024, n_gpu_layers = 1, n_batch = 4)
            
        elif self.llm_model_name == 'ggml-falcon':
            self.llm = GPT4All(model = r"D:\Downloads\ggml-model-gpt4all-falcon-q4_0.bin")
            # verbose = True, n_ctx = 1024, n_gpu_layers = 1, n_batch = 4)
            
        elif self.llm_model_name.startswith('flan'):
            tokenizer = AutoTokenizer.from_pretrained(f"google/{self.llm_model_name}")
            model = AutoModelForSeq2SeqLM.from_pretrained(f"google/{self.llm_model_name}")
            pipe = pipeline("text2text-generation", model = model, tokenizer = tokenizer)
            self.llm = HuggingFacePipeline(
                pipeline = pipe,
                model_kwargs = {"temperature": 0, "max_length": 512},
            )
            
        elif self.llm_model_name.startswith('distilbert'):
            tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/distilbert-base-nli-stsb-mean-tokens")
            model = AutoModelForSeq2SeqLM.from_pretrained("sentence-transformers/distilbert-base-nli-stsb-mean-tokens")
            pipe = pipeline("text2text-generation", model = model, tokenizer = tokenizer)
            self.llm = HuggingFacePipeline(
                pipeline = pipe,
            )
            
        elif self.llm_model_name.startswith('bert'):
            tokenizer = AutoTokenizer.from_pretrained(f"sentence-transformers/bert-base-nli-stsb-mean-tokens")
            model = AutoModelForSeq2SeqLM.from_pretrained("sentence-transformers/bert-base-nli-stsb-mean-tokens")
            pipe = pipeline("text2text-generation", model = model, tokenizer = tokenizer)
            self.llm = HuggingFacePipeline(
                pipeline = pipe,
            )
            
        elif self.llm_model_name.startswith('roberta'):
            tokenizer = AutoTokenizer.from_pretrained(f"deepset/roberta-base-squad2")
            model = RobertaForCausalLM.from_pretrained("deepset/roberta-base-squad2")
            pipe = pipeline("text2text-generation", model = model, tokenizer = tokenizer)
            self.llm = HuggingFacePipeline(
                pipeline = pipe,
            )
        
        if not self.llm:
            raise NameError("The model_name for llm that you entered is not supported")
    
    def llm_rag(self,
                query: str,
                db):
        """
        Performs Retrieval Augmented Generation with the most similar document from the vector db
        """
        
        query = query.lower()
        
        result = None
        answer = 'not contain the answer'
        current_k = 0
        while 'not contain the answer' in answer and current_k <= 1:
            current_k += 1
            qa = RetrievalQA.from_chain_type(llm = self.llm,
                                             chain_type = "stuff",
                                             retriever = db.as_retriever(search_kwargs = {'k': current_k}),
                                             chain_type_kwargs = {"prompt": self.llm_rag_prompt},
                                             return_source_documents = True
                                             )
            result = qa({"query": query})
            answer = result['result']
        
        # console_print(result, 'result')
        relevant_docs, similarity_scores = self.relevant_docs_ordered_by_similarity(query, db, current_k)
        # console_print(relevant_docs, 'relevant_docs')
        return result, relevant_docs
    
    @staticmethod
    def relevant_docs_ordered_by_similarity(query: str,
                                            db,
                                            k: int,
                                            threshold: float = 0.5):
        """
        Returns the most similar documents to the query depending on a similarity threshold
        """
        
        relevant_docs_tuples = db.similarity_search_with_relevance_scores(query, k = k)
        
        # sort by relevance score
        relevant_docs_tuples.sort(key = lambda a: a[1], reverse = True)
        
        # take only relevant docs with cosine similarity > 0.5
        relevant_docs = [pair[0] for pair in relevant_docs_tuples if pair[1] >= threshold]
        similarity_scores = [pair[1] for pair in relevant_docs_tuples if pair[1] >= threshold]
        
        console_print('Most similar documents')
        for i in range(len(relevant_docs)):
            console_print(f'{similarity_scores[i]} {relevant_docs[i]}')
        
        return relevant_docs, similarity_scores


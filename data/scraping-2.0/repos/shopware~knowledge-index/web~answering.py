from web.config import get_embedding_fn, db_dir, data_dir, sqlite_dir
from web.vector_store import FaissMap

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA, LLMChain, ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.schema.runnable import RunnablePassthrough
from langchain.callbacks import get_openai_callback
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache

from .tracking import send_event

import os
import hashlib
from pathlib import Path

async def generate_answer(question: str, collection):
    my_db_dir = db_dir(collection)
    my_data_dir = data_dir(collection)

    search_index = FaissMap.load_local(my_db_dir, get_embedding_fn())
    
    answeringFactory = AnsweringFactory()
    modelFactory = ModelFactory()
    output = None

    # https://python.langchain.com/docs/modules/model_io/llms/token_usage_tracking
    cb = get_openai_callback()

    # https://python.langchain.com/docs/integrations/llms/llm_caching
    # create cache dir
    my_sqlite_dir = sqlite_dir(collection)
    if not os.path.isdir(my_sqlite_dir):
        Path(my_sqlite_dir).mkdir(exist_ok=True, parents=True)
    set_llm_cache(SQLiteCache(database_path=Path(my_sqlite_dir) / ".langchain.db"))

    instances = 0

    with get_openai_callback() as cb:
        if True:
            instances = instances + 1
            mode = 'stuffedprompt'
            #mode = 'noprompt'
            instance = answeringFactory.create(mode, search_index, "gpt-3.5-turbo")

            output = instance.reformat(instance.run(question), my_data_dir)
        else:
            questions = [
                #"Help me create an app.",
                #"How can I aggregate products by category?",
                #"How can I create custom CMS elemnts?",
                #"How can I extend Shopware?",
                #"How do I create a new controller?",
                "List me the slots of the SW-Card",
                "What icons can I use to represent the shopping cart?",
                "What is the code for adding a primary button?",
                #"What's the authentication for the ADmin API based on?",
                #"Where can I download Shopware?",
            ]

            results = {}
            for mode in answeringFactory.getMapper():
                for q in questions:
                    for model in modelFactory.getMapper():
                        try:
                            instances = instances + 1
                            # print("Matrix Q: " + q + " Model: " + model + " Mode: " + mode)
                            instance = answeringFactory.create(mode, search_index, model)

                            response = instance.reformat(instance.run(q), my_data_dir)
                            response["question"] = q
                            results[mode + ":" + model + ":" + hashlib.sha1(q.encode('utf-8')).hexdigest()] = response
                        except Exception as e:
                            results[mode + ":" + model + ":" + hashlib.sha1(q.encode('utf-8')).hexdigest()] = str(e)
                            print(e)

            output = results

        # track event
        await send_event('all', 'qa', {**{"question": question, "collection": collection, "instances": instances}, **cb.__dict__})

        output['stats'] = cb
    
    return output


# START MODELS
class ModelInterface:
    def __init__(self):
        pass
    
    def getContext(self):
        return self.context
    
    def getLLM(self):
        # https://python.langchain.com/docs/use_cases/question_answering/vector_db_qa
        max_tokens = 1024 # 512
        max_tokens = -1
        #batch_size = 5
        
        self.llm = OpenAI(
            temperature=0.0,
            max_tokens=max_tokens,
            model_name=self.name,
            #batch_size=batch_size
        )

        return self.llm

class ModelFactory:

    def getMapper(self):
        return {
            # "gpt-4-1106-preview": GPT41106Preview, # not for production
            # "gpt-4": GPT4, # "I don't know"
            #"gpt-4-32k": GPT432k, # no access
            # "gpt-3.5-turbo-1106": GPT35Turbo1106, # weird results
            "gpt-3.5-turbo": GPT35Turbo,
            #"gpt-3.5-turbo-instruct": GPT35TurboInstruct
        }

    def create(self, name: str) -> ModelInterface:
        mapping = self.getMapper()

        selected_class = mapping.get(name)

        if selected_class:
            return selected_class()
        
        raise ValueError("Incorrect model implementation. Available: " + ','.join(mapping.keys()))

class GPT41106Preview(ModelInterface):
    name = "gpt-4-1106-preview"
    context = 128000

class GPT4(ModelInterface):
    name = "gpt-4"
    context = 8192

class GPT432k(ModelInterface):
    name = "gpt-4-32k"
    context = 32768

class GPT35Turbo1106(ModelInterface):
    name = "gpt-3.5-turbo-1106"
    context = 16385

class GPT35Turbo(ModelInterface):
    name = "gpt-3.5-turbo"
    context = 4096

class GPT35TurboInstruct(ModelInterface):
    name = "gpt-3.5-turbo-instruct"
    context = 4096

# START ANSWERS
class AnsweringInterface:
    llm = None

    def __init__(self, search_index, model):
        self.search_index = search_index
        self.model = model

    def run(self, question: str):
        pass

    def meta(self):
        pass
    
    def getLLM(self):
        if (self.llm):
            return self.llm

        modelFactory = ModelFactory()

        self.llm = modelFactory.create(self.model).getLLM()

        return self.llm

    def getRetriever(self):
        modelFactory = ModelFactory()

        return self.search_index.as_retriever(
            search_kwargs={
                'k': 15,
                'filter': {
                    'version': 'latest'
                },
                'model': modelFactory.create(self.model),
            }
        )
    
    def getDocuments(self, question: str):
        return self.getRetriever().get_relevant_documents(question)

    def splitAnswerAndSources(self, answer):
        split = answer.split('\nSOURCES: ')

        return {
            'answer': split[0],
            'sources': split[1],
        }

    def mapSourcesFromDocuments(self, documents):
        sources = []
        for document in documents:
            sources.append(document.metadata["source"])

        return sources

    def reformat(self, output, my_data_dir):
        if 'sources' not in output:
            output['sources'] = []
        elif output['sources'] == 'None.':
            output['sources'] = []
        elif not isinstance(output['sources'], list):
            output['sources'] = output['sources'].split(', ')

        # normalize sources
        #if type(output['sources']) != list:
        output['sources'] = [s[len(my_data_dir):] for s in output['sources']]

        return output


class AnsweringFactory:
    def getMapper(self):
        return {
            #'noprompt': NoPrompt,
            'stuffedprompt': StuffedPrompt,
            #'ragchain': RagChain,
            #'mapreduce': MapReduce,
            #'italian': Italian,
            #'reducedprompt': ReducedPrompt,
            ## 'testchat': TestChat,
            #'anothertestchat': AnotherTestChat,
            #'chatbot': Chatbot,
        }

    def create(self, name: str, search_index, model: str) -> AnsweringInterface:
        mapping = self.getMapper()

        selected_class = mapping.get(name)

        if selected_class:
            return selected_class(search_index, model)
        
        raise ValueError("Incorrect answering implementation. Available: " + ','.join(mapping.keys()))


class NoPrompt(AnsweringInterface):
    def run(self, question: str):
        chain = RetrievalQAWithSourcesChain.from_chain_type(
            self.getLLM(),
            chain_type="map_reduce",
            retriever=self.getRetriever(),
            verbose = True,
        )

        return chain({"question": question}, return_only_outputs=True)
    
    def meta(self):
        return {
            "prompt": None,
            "chain": "RetrievalQAWithSourcesChain",
            "type": "map_reduce",
        }

class StuffedPrompt(AnsweringInterface):
    def run(self, question: str):
        prompt_template = """
You are an AI assistant for the developer documentation of the eCommerce API Backend Shopware. The documentation is located at https://developer.shopware.com.
You are given extracted parts of a long document for context and a question to answer.
If the question includes a request for code, provide a code block directly from the documentation.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not about Shopware, politely inform them that you are tuned to only answer questions about Shopware.
Wrap single-line code or keyword into single tick (`). Wrap any code into triple ticks (```).

Context:
{context}

Question:
{question}

Answer in Markdown:
"""

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        # chain = LLMChain(llm=llm, prompt=PROMPT)
        chain = RetrievalQA.from_chain_type(
            self.getLLM(),
            chain_type="stuff",
            retriever=self.getRetriever(),
            chain_type_kwargs = {"prompt": PROMPT},
            return_source_documents=True
        )

        result = chain(question, return_only_outputs=True)

        return {
            'answer': result['result'],
            'sources': self.mapSourcesFromDocuments(result["source_documents"]),
        }


class RagChain(AnsweringInterface):
    def run(self, question: str):
        # https://python.langchain.com/docs/use_cases/question_answering/
        template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer. 
        Use three sentences maximum and keep the answer as concise as possible.
        Transform code to markdown.
        {context}
        Question: {question}
        Helpful Answer:"""
        rag_prompt_custom = PromptTemplate.from_template(template)

        rag_chain = (
            {"context": self.getRetriever(), "question": RunnablePassthrough()} | rag_prompt_custom | self.getLLM()
        )

        return {
            'answer': rag_chain.invoke(question),
            'sources': self.mapSourcesFromDocuments(self.getDocuments(question))
        }

class MapReduce(AnsweringInterface):
    def run(self, question: str):
        # https://python.langchain.com/docs/use_cases/question_answering/in_memory_question_answering
        chain = load_qa_chain(self.getLLM(), chain_type="map_reduce")
        docs = self.getDocuments(question)
        
        result = chain({"input_documents": docs, "question": question}, return_only_outputs=True)

        return {
            'answer': result['output_text'],
            'sources': self.mapSourcesFromDocuments(docs),
        }

class Italian(AnsweringInterface):
    def run(self, question: str):
        question_prompt_template = """Use the following portion of a long document to see if any of the text is relevant to answer the question.
        Return any relevant text translated into italian.
        {context}
        Question: {question}
        Relevant text, if any, in Italian:"""
        QUESTION_PROMPT = PromptTemplate(
            template=question_prompt_template, input_variables=["context", "question"]
        )

        combine_prompt_template = """Given the following extracted parts of a long document and a question, create a final answer italian.
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.

        QUESTION: {question}
        =========
        {summaries}
        =========
        Answer in Italian:"""
        COMBINE_PROMPT = PromptTemplate(
            template=combine_prompt_template, input_variables=["summaries", "question"]
        )
        chain = load_qa_chain(self.getLLM(), chain_type="map_reduce", return_map_steps=True, question_prompt=QUESTION_PROMPT, combine_prompt=COMBINE_PROMPT)
        docs = self.getDocuments(question)

        result = chain({"input_documents": docs, "question": question}, return_only_outputs=True)

        return {
            'answer': result['output_text'],
            'sources': self.mapSourcesFromDocuments(docs),
        }

class ReducedPrompt(AnsweringInterface):
    def run(self, question: str):
        question_prompt_template = """Use the following portion of a long document to see if any of the text is relevant to answer the question.
        Return any relevant text.
        {context}
        Question: {question}
        Relevant text:"""
        QUESTION_PROMPT = PromptTemplate(
            template=question_prompt_template, input_variables=["context", "question"]
        )

        combine_prompt_template = """Given the following extracted parts of a long document and a question, create a final answer.
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.

        QUESTION: {question}
        =========
        {summaries}
        =========
        Answer:"""
        COMBINE_PROMPT = PromptTemplate(
            template=combine_prompt_template, input_variables=["summaries", "question"]
        )
        chain = load_qa_chain(self.getLLM(), chain_type="map_reduce", return_map_steps=True, question_prompt=QUESTION_PROMPT, combine_prompt=COMBINE_PROMPT)
        docs = self.getDocuments(question)

        result = chain({"input_documents": docs, "question": question}, return_only_outputs=True)

        return {
            'answer': result['output_text'],
            'sources': self.mapSourcesFromDocuments(docs),
        }

class TestChat(AnsweringInterface):
    def run(self, question: str):
        # NOT WORKING, extra fields not permitted
        # https://github.com/langchain-ai/langchain/issues/5096
        combine_template = "Write a summary of the following text:\n\n{summaries}"
        combine_prompt_template = PromptTemplate.from_template(template=combine_template)

        #question_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
        #{context}
        #Question: {question}
        #Helpful Answer:"""
        #question_prompt_template = PromptTemplate.from_template(template=question_template)

        # create retriever chain
        qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=self.getLLM(),
            # mmr > for diversity in documents
            # Set fetch_k value to get the fetch_k most similar search. This is basically semantic search
            retriever=self.getRetriever(),
            return_source_documents=True,
            chain_type="map_reduce",
            #chain_type_kwargs={"question_prompt": question_prompt_template, "combine_prompt": combine_prompt_template}
            chain_type_kwargs={"prompt": combine_prompt_template}
        )

        # call QA chain
        return qa_chain({"query": question})

class AnotherTestChat(AnsweringInterface):
    def run(self, question: str):
        qa_chain = load_qa_chain(self.getLLM(), chain_type="stuff")
        qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=self.getRetriever())

        return {
            'answer': qa.run(question),
            'sources': self.mapSourcesFromDocuments(self.getDocuments(question)),
        }

class Chatbot(AnsweringInterface):
    def run(self, question: str):
        #chatbot = ConversationalRetrievalChain.from_llm(
        #            llm=llm, 
        #           condense_question_llm=ChatOpenAI(temperature=0, model='gpt-3.5-turbo'),
        #            retriever=retriever,
        #            chain_type="map_reduce",
        #            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True), 
        #           verbose=True,
        #            #return_generated_question=<BOOL>,
        #            #get_chat_history=lambda h : h, 
        #            return_source_documents=True
        #)

        question_generator = LLMChain(llm=self.getLLM(), prompt=CONDENSE_QUESTION_PROMPT)
        #doc_chain = load_qa_chain(llm, chain_type="map_reduce")
        doc_chain = load_qa_with_sources_chain(self.getLLM(), chain_type="map_reduce")

        chatbot = ConversationalRetrievalChain(
            retriever=self.getRetriever(),
            question_generator=question_generator,
            combine_docs_chain=doc_chain,
            return_source_documents=True
        )

        chat_history = []
        result = chatbot({"question": question, "chat_history": chat_history})

        return {
            "answer": self.splitAnswerAndSources(result['answer'])["answer"],
            "sources": self.mapSourcesFromDocuments(result["source_documents"]),
        }
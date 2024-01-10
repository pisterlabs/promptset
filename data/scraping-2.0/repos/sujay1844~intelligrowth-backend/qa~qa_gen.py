from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.chains import RetrievalQA
from langchain.callbacks import StdOutCallbackHandler

from model.model_loader import model, tokenizer
from database.data_loader import esops_documents
from database.data_loader import data_root
from database.chroma import vector_db
from prompts.qa_prompt import qa_prompt, questions_prompt, answer_prompt

qa_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=2048,
    do_sample=True,
    temperature=0.1,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1
)

llm = HuggingFacePipeline(pipeline=qa_pipeline)

class QAGenerator:
    def __init__(self, number, topics):
        self.number = number
        sources = [f'{data_root}/{topic}.txt' for topic in topics]
        handler = StdOutCallbackHandler()
        bm25_retriever = BM25Retriever.from_documents(esops_documents)
        bm25_retriever.k=5
        chroma_retriever=vector_db.as_retriever(
            search_kwargs={"k":5},
            filter={"source":sources}
        )
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever,chroma_retriever],
            weights=[0.5,0.5]
        )

        self.qa_with_sources_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=ensemble_retriever,
            verbose=True,
            callbacks=[handler],
            chain_type_kwargs={"prompt": qa_prompt},
            return_source_documents=True
        )
        self.topics_list = [topic.replace(".txt", "").replace(f"{data_root}/", "") for topic in topics]
    
    def get_questions(self):
        q_query = questions_prompt.format(number=self.number, topics_list=self.topics_list)

        self.questions = self.qa_with_sources_chain({'query':q_query})['result'].split("\n")

        return self.questions
    
    def get_answers(self):
        if self.questions == None:
            raise Exception("Questions not generated")

        a_query = answer_prompt.format(questions=self.questions)

        self.answers = self.qa_with_sources_chain({"query":a_query})['result'].split("\n")

        return self.answers

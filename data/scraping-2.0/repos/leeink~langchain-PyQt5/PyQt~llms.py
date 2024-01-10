import os
import glob

# openAI API key
os.environ['OPENAI_API_KEY']='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'

from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.indexes import VectorstoreIndexCreator


# LLM을 이용한 문서 요약기
class summarizer:
    def __init__(self):
        self.llm = OpenAI(temperature=0)
        self.text_splitter = CharacterTextSplitter()
        self.chain_type = "map_reduce"
        self.chain = load_summarize_chain(self.llm, chain_type=self.chain_type)
        self.map_template = """Give the following python code information, generate a description that briefly explains what the code does.

        Return the the description in the following format:
        description of the function
        {code}
        """
        self.reduce_template = """Give the following following fuctions name and their descritpion,
                                 answer the following question
                                {code_description}
                                Question: {question}
                                Answer:
                                """
        self.MAP_PROMPT = PromptTemplate(input_variables=["code"], template=self.map_template)
        self.REDUCE_PROMPT = PromptTemplate(input_variables=["code_description", "question"], template=self.reduce_template)
        self.map_llm_chain = LLMChain(llm=self.llm, prompt=self.MAP_PROMPT)
        self.reduce_llm_chain = LLMChain(llm=self.llm, prompt=self.REDUCE_PROMPT)
        self.generative_result_reduce_chain = StuffDocumentsChain(
                            llm_chain=self.reduce_llm_chain,
                            document_variable_name="code_description",)
        self.combine_documents = MapReduceDocumentsChain(
                            llm_chain=self.map_llm_chain,
                            combine_document_chain=self.generative_result_reduce_chain,
                            document_variable_name="code",)
        self.map_reduce = MapReduceChain(combine_documents_chain=self.combine_documents,
                            text_splitter=CharacterTextSplitter(separator="\n##\n", chunk_size=100, chunk_overlap=0),)

    
    # TXT문서의 내용을 요약하는 함수
    def summary(self, doc):
        texts = self.text_splitter.split_text(doc)
        docs = [Document(page_content=t) for t in texts[:3]]
        return self.chain.run(docs)
    
    # PDF문서 내용을 요약하는 함수
    def summaryPdf(self, path, custom_prompt=""):
        loder = PyPDFLoader(path)
        docs = loder.load_and_split()
        chain = load_summarize_chain(self.llm, chain_type=self.chain_type)
        summary = chain.run(docs)
        if custom_prompt !="":
            prompt_template = custom_prompt +"""

            {text}
                
            SUMMARY:"""
            PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
            chain = load_summarize_chain(self.llm, chain_type=self.chain_type,
                                                  map_prompt=PROMPT, combine_prompt=PROMPT)
            custom_summary = chain({"input_documents": docs}, return_only_outputs=True)["output_text"]

        else:
            custom_summary = ""

        return summary, custom_summary
    
    # 미사용 함수
    def custom_summary(self,path,custom_prompt):
        loader = PyPDFLoader(path)
        docs = loader.load_and_split()
        prompt_template = custom_prompt +"""
        {text}
        SUMMARY:"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
        chain = load_summarize_chain(self.llm, chain_type=self.chain_type,
                                                    map_prompt=PROMPT, combine_prompt=PROMPT)
        summary_output = chain({"input_documents": docs}, return_only_outputs=True)["output_text"]

        return summary_output
    
    # 불러온 폴더 내에 PDF문서들을 요약하는 함수
    def summarize_pdfs_from_folder(self,pdfs_folder):
        summarize =[]
        for pdf in glob.glob(pdfs_folder + "/*.pdf"):
            loader = PyPDFLoader(pdf)
            docs = loader.load_and_split()
            chain = load_summarize_chain(self.llm, chain_type=self.chain_type)
            summary = chain.run(docs)
            summarize.append(summary)

        return summarize
    
    # 불러온 폴더 내 PDF문서 내 내용에 대해 질문을 보내는 함수
    def QueryPdfs_fromFolder(self,pdfs_folder,query):
        loader = PyPDFDirectoryLoader(pdfs_folder)
        #docs = loader.load()
        index = VectorstoreIndexCreator().from_loaders([loader])
        res = index.query(query)
        return res
    
    # 코드를 요약하는 함수
    def summarize_code(self, code, query):
        return self.map_reduce.run(input_text = code, question = query)
    
class gererator:
    def __init__(self):
        self.llm = OpenAI(temperature=0)
        self.text_splitter = CharacterTextSplitter()
        self.chain_type = "map_reduce"
        self.chain = load_summarize_chain(self.llm, chain_type=self.chain_type)
        self.map_template = """
        If the given information is a description that explains the task of a given Python code, create the corresponding description.
        If it is a request for code generation based on the provided information, generate the code accordingly.
        {input_text}
        """
        self.reduce_template = """Generate the code according to the given question.
                                {question}
                                """
        self.MAP_PROMPT = PromptTemplate(input_variables=["input_text"], template=self.map_template)
        self.REDUCE_PROMPT = PromptTemplate(input_variables=["question"], template=self.reduce_template)
        self.map_llm_chain = LLMChain(llm=self.llm, prompt=self.MAP_PROMPT)
        self.reduce_llm_chain = LLMChain(llm=self.llm, prompt=self.REDUCE_PROMPT)
        self.generative_result_reduce_chain = StuffDocumentsChain(
                            llm_chain=self.reduce_llm_chain,
                            document_variable_name="question",)
        self.combine_documents = MapReduceDocumentsChain(
                            llm_chain=self.map_llm_chain,
                            combine_document_chain=self.generative_result_reduce_chain,
                            document_variable_name="input_text",)
        self.map_reduce = MapReduceChain(combine_documents_chain=self.combine_documents,
                            text_splitter=CharacterTextSplitter(separator="\n##\n", chunk_size=100, chunk_overlap=0),)

        
#g = gererator()
#llm = OpenAI(temperature=0)
#print(llm.predict("def shell_sort(arr):"))
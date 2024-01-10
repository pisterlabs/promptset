from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

from langchain.vectorstores import FAISS
from langchain.schema import Document

from langchain.text_splitter import RecursiveCharacterTextSplitter

import PyPDF2

"""f
Chain: Read all text --> extract all tables using OpenAI --> extract all text using OpenAI -->
--> use every row as a chunk ---> use every paragraphs as a junk --> embedd each of them

1. I will read all the text using PyPDF2
2_1. I will extract all tables for each page using a PromptTemplate and asking OpenAI to extract
2_2. Extract all graphs
3. I will itterate over every individual row and chunk that together, using "---" as a separator
   I will use recursiveTextSplitter to split the text
4. I will embed using OpenAI
5. Use SQL-database to store embeddings, add a tag for PDF_embedding_title

// there are no need for user agents
// incorporate GoogleSearchAPIWrapper
"""


class Read_pdf:
    def __init__(self):
        self.db = None

    def get_all_text(self, pdf_title): #FUNKER
        text = {}

        with open(pdf_title, "rb") as PDF:
            reader = PyPDF2.PdfReader(PDF)

            if reader.is_encrypted:
                reader.decrypt("")

            num_of_pages = len(reader.pages)

            for page_num in range(num_of_pages):
                page = reader.pages[page_num]
                text[page_num] = page.extract_text()

        return text
    
    def embed_text(self, texts): #texts={page_num, text}
        self.embedder = OpenAIEmbeddings()
        
        #create a Document for each page instead of plain text
        docs = []
        for page_num in texts.keys():
            #breaking up every text into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
            chunks = splitter.split_text(text=texts[page_num],)

            for chunk in chunks:

                doc = Document(page_content=chunk, metadata= {"page_num": page_num})
                docs.append(doc)

        #create vectorstore, every page should be one vector
        self.db = FAISS.from_documents(docs, self.embedder)

        return self.db
    

    def perform_similarity_search(self, query):
        docs = FAISS.similarity_search(self.db, query)
        return docs

    
    
    def query_data(self, query, docs): #query="Whats the weather", docs=Docs from similaruty search
        llm = OpenAI(model="text-davinci-003", temperature=0)

        content = ""
        for doc in docs:
            content += doc.page_content
            content += "\n"


        template = PromptTemplate(
            input_variables=["content", "query"],
            template="""Answer politely the following query using available data. If you do not know the answer, say so.
                        data: {content}
                        query: {query}"""
        )


        chain = LLMChain(llm=llm, prompt=template)

        response = chain(inputs={"content": content, "query": query})
        return response, docs #docs will be used to gather page_nums
    

    def upload_and_create_vectorstore(self, pdf_title):
        texts = self.get_all_text(pdf_title=pdf_title)
        self.db = self.embed_text(texts=texts)
    
    def main(self, pdf_title, query):
        if self.db == None:
            self.upload_and_create_vectorstore(pdf_title=pdf_title)

        response, relevant_docs = self.query_data(query=query, docs=self.perform_similarity_search(query=query))

        return response, relevant_docs

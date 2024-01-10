import os  # Provides functions for interacting with the operating system (Checks for a Path)

from dotenv import load_dotenv
from PyPDF2 import PdfReader # Allows reading PDF files
import pickle
import torch


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain



# def main():
#     pdf_path="GAN.pdf"
#     store_name = "GAN"
#     #Add this to the query
#     #I also want only 2 sentences
#     query =  "Why study generative modeling?"
#     # Limit the maximum number of results to 10
#     max_num_results = 10

#     #rb = binary read mode
#     #rb is needed for pdf files (non text files)
#     with open(pdf_path, "rb") as pdf_file:
#         #read contents in the pdf file
#         pdf_reader = PdfReader(pdf_file)
#         text = ""
#         for page in pdf_reader.pages:
#             text+=page.extract_text()
#         #places the text in the pdf file in the variable text

#         #Splits text
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=800,
#             chunk_overlap=100,
#             length_function=len
#         )

#         chunks = text_splitter.split_text(text=text)


#         #turns text into chunks 
    
#     #If the pk1 file is already created
#     #If the serializated Vector Store is completed
#     if os.path.exists(f"{store_name}.pk1"):
#         #open as a binary file (pk1)
#         with open(f"{store_name}.pk1", "rb") as f:
#             #loads the vectorStore objects from the pk1 file
#             VectorStore = pickle.load(f)

        
                    
#     #If the pk1 file needs to be created
#     else:
#         #Create embedding of the text chunks
#         embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

#         #Creates a VectorStore Object
#         VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
#         # with open(f"{store_name}.pdf", "rb") as f:
#         #     pickle.dump(VectorStore, f)

#         #Places the VectorStore object in the pk1 file
#         with open(f"{store_name}.pk1", "wb") as f:  # Use "wb" for binary writing
#             pickle.dump(VectorStore, f)

  


#     #Loads the VectorStore from the pk1 file
#     with open(f"{store_name}.pk1", "rb") as f:
#          VectorStore = pickle.load(f)




#     #If the query exists
#     if query:
#         #Returns chunks that match the query to some extent
#         docs = VectorStore.similarity_search(query=query, num_results=max_num_results)
#         #print("Documents retrieved for query:", docs)

#         #temperature = creativity and random 2 is the max ( 2- temperature)
#         #llm is a language model

#         #stop = "." means we get only 1 sentence
#         #max_tokens = 8 means max number of words 
#         llm = OpenAI(openai_api_key=openai_api_key,temperature=0, frequency_penalty= 1.0)

#         #chain holds the loaded question answering chain
#         #stuff for basic answer
#         #refine for a more complex answer
#         chain = load_qa_chain(llm=llm, chain_type="stuff")

#         #run the chain
#         #input_documents provides relevant information
#         response = chain.run(input_documents=docs, question=query)

#     return response

class PDFSCanner:

    def createEmbedding(self,pdf_path,store_name):

        pdf_path= pdf_path
        store_name = store_name

        #rb = binary read mode
        #rb is needed for pdf files (non text files)
        with open(pdf_path, "rb") as pdf_file:
            #read contents in the pdf file
            pdf_reader = PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text+=page.extract_text()
            #places the text in the pdf file in the variable text

            #Splits text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                length_function=len
            )

            chunks = text_splitter.split_text(text=text)


            #turns text into chunks 
        
        #If the pk1 file is already created
        #If the serializated Vector Store is completed
        if os.path.exists(f"{store_name}.pk1"):
            #open as a binary file (pk1)
            with open(f"{store_name}.pk1", "rb") as f:
                #loads the vectorStore objects from the pk1 file
                VectorStore = pickle.load(f)

            
                        
        #If the pk1 file needs to be created
        else:
            #Create embedding of the text chunks
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

            #Creates a VectorStore Object
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            # with open(f"{store_name}.pdf", "rb") as f:
            #     pickle.dump(VectorStore, f)

            #Places the VectorStore object in the pk1 file
            with open(f"{store_name}.pk1", "wb") as f:  # Use "wb" for binary writing
                pickle.dump(VectorStore, f)

    


        #Loads the VectorStore from the pk1 file
        with open(f"{store_name}.pk1", "rb") as f:
            VectorStore = pickle.load(f)


        return VectorStore

    def generateResponse (self,VectorStore,openai_api_key,query,isSimpleQuestion,isCreative, repetitiveAnswer) :

        #If the query exists
        if query:
            #Returns chunks that match the query to some extent
            docs = VectorStore.similarity_search(query=query, num_results=isSimpleQuestion)
            #print("Documents retrieved for query:", docs)

            #temperature = creativity and random 2 is the max ( 2- temperature)
            #llm is a language model

            #stop = "." means we get only 1 sentence
            #max_tokens = 8 means max number of words 
            llm = OpenAI(openai_api_key=openai_api_key,temperature=isCreative, frequency_penalty= repetitiveAnswer)

            #chain holds the loaded question answering chain
            #stuff for basic answer
            #refine for a more complex answer
            chain = load_qa_chain(llm=llm, chain_type="stuff")

            #run the chain
            #input_documents provides relevant information
            response = chain.run(input_documents=docs, question=query)

        return response

#main()
def main () :
    pdf_path="GAN.pdf"
    store_name = "GAN"
    pk1File = "GAN.pk1"
    #I also want only 2 sentences
    query =  "Why study generative modeling?"
    openai_api_key = ""
    # Limit the maximum number of results to 10
    max_num_results = 10
    #5 means simple, 15 means complex
    isSimpleQuestion = 5


    #Not Creative
    isCreative = 0
    #1 means not repetiive, 0 means repetitive
    repetitiveAnswer = 1.0

    pdfScanner = PDFSCanner()

    VectorStore = pdfScanner.createEmbedding(pdf_path,store_name)
    response = pdfScanner.generateResponse(VectorStore, openai_api_key, query,isSimpleQuestion, isCreative, repetitiveAnswer)
    print(response)



if __name__ == "__main__":
    main()






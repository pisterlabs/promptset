from dotenv import load_dotenv, find_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback


def main():
#    import pdb; pdb.set_trace()
    dotenv_file = find_dotenv(usecwd=True)
    print(dotenv_file)
    load_dotenv(dotenv_file)
    #load_dotenv()
    import os
    print(os.environ)
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")
    
    # upload file
#    pdf_files = st.file_uploader("Upload your PDFs", accept_multiple_files=True)
    pdf_files = ["./ad.pdf"]
    print(pdf_files)
#    pdf_files = st.file_uploader("Upload your PDFs", type="pdf" )
    
    # extract the text
    if pdf_files is not None:
      text = ""
      for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
          text += page.extract_text()
        
      # split into chunks
      text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
      chunks = text_splitter.split_text(text)
#      print(chunks)
      
#      import pdb; pdb.set_trace()

      # create embeddings
      embeddings = OpenAIEmbeddings()
      #print("Embed:*******************************",len(embeddings))
      print("Embed:*******************************")
      knowledge_base = FAISS.from_texts(chunks, embeddings)
      
      # show user input
      #user_question = st.text_input("Ask a question about your PDF:")
      user_question = "As per Youtube Content Guidelines, what are Swear words? And does the transcript below can get penalised - 'i0:06 well welcome guys to my office or second 0:10 food whatever you wanna call it so come 0:13 in it is more of an office with this 0:16 of the people with giant spend a 0:18 lot on it is a big people it has a lot 0:21 of it does not fuck fuck have other storage but 0:23 the cool thing is that this is movable' " 
      if user_question:
#        docs = knowledge_base.similarity_search(user_question)
        docs = knowledge_base.similarity_search_with_score(user_question)
        docnew=[]
        for d in docs:
          if d is not None:
              if hasattr(d,'page_content'):
                docnew.append(d)
          print("---")
        
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        #chain = load_qa_chain(llm, chain_type="map_reduce")
        with get_openai_callback() as cb:
          #response = chain.run(input_documents=docs, question=user_question)
          response = chain.run(input_documents=docnew, question=user_question)
          print(response)
           
        #st.write(response)
    

if __name__ == '__main__':
    main()

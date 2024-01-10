#import libraries
import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain import hub
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv
import sys

sys.path.append("src")
import arxiv_retriever
import pubmed_retriever
import create_docs

# Load the environment variables
load_dotenv()

# Initialize session state variables
def initialize_state():
    if "articles_list" not in st.session_state:
        st.session_state["articles_list"] = None
    if "documents" not in st.session_state:
        st.session_state["documents"] = None
    if "vectorstore" not in st.session_state:
        st.session_state["vectorstore"] = None


def reset_state():
    for key in st.session_state.keys():
        del st.session_state[key]
    initialize_state()

def get_vectorstore(documents):
    return Chroma.from_documents(documents=documents, embedding=OpenAIEmbeddings())

# Main function
def main():
    st.title("Research Paper Explorer")
    initialize_state()

    with st.sidebar:
        st.header("Search")
        source = st.radio("Select your source:", ('arXiv', 'PubMed'))
        num_of_articles = st.slider('Number search results', 0, 100, 10)
        search_term = st.text_input("Enter Search Term:")
        if st.button("Retrieve"):
            reset_state()  
            if search_term:
                try:
                    with st.spinner("...loading..."):
                        if source == 'arXiv':
                            st.session_state.articles_list = arxiv_retriever.arxiv_search(search_term, num_of_articles)
                        elif source == 'PubMed':
                            st.session_state.articles_list = pubmed_retriever.pubmed_search(search_term, num_of_articles)  

                        st.session_state.documents = create_docs.create_documents(st.session_state.articles_list)
                        st.session_state.vectorstore = get_vectorstore(st.session_state.documents)
                except Exception as e:
                    st.error(f"An error occurred: {e}")  

    if search_term == "":
        st.write("This app retrieves abstracts from the arXiv API and uses them to answer questions and return pdf-Links as reference. Enter a search term in the sidebar to get started.")

    if search_term and st.session_state.vectorstore:
        # Prepare the RAG chain
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 10})
        llm = ChatOpenAI(model_name="gpt-3.5-turbo")
        prompt = ChatPromptTemplate(input_variables=['context', 'question'], 
                                    messages=[HumanMessagePromptTemplate(
                                        prompt=PromptTemplate(
                                            input_variables=['context', 'question'], 
                                            template="""Use the following pieces of context to answer the question at the end.
                                            If you don't know the answer, just say that you don't know, don't try to make up an answer.\n
                                            Question: {question} \nContext: {context} \nAnswer:"""))])

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
            | prompt
            | llm
            | StrOutputParser()
        )

        rag_chain_with_source = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)

        # Streamlit input for user question
        user_question = st.text_input("Enter your question about the research papers here:")
        if st.button("Ask"):
            if user_question:
                output = rag_chain_with_source.invoke(user_question)

                # Display the answer
                st.write("Answer:", output["answer"])
                
                # Displaying PDF URLs as Sources
                titles_and_pdf_urls = [(doc.metadata['title'], doc.metadata['link']) for doc in output['context']]
                titles_and_pdf_urls_string = "  \n".join([f"{title}: {url}" for title, url in titles_and_pdf_urls])
                st.write("Titles and PDF URLs:  \n{}".format(titles_and_pdf_urls_string))
            else:
                st.write("Please enter a question and press button to retrieve answer.")

    else:
        st.write("Please enter a search term in the sidebar and press button to retrieve results.")        

if __name__ == "__main__":
    main()

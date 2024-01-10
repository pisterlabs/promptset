from langchain.document_loaders import UnstructuredCSVLoader
from langchain.chat_models import ChatOpenAI
from langchain.indexes import VectorstoreIndexCreator
import streamlit as st
import os


# Load the model and data
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
data = UnstructuredCSVLoader('final.csv')
index = VectorstoreIndexCreator().from_loaders([data])


def main():
    st.title("LangChain Chat App")
    st.write("Enter your query in the text box below and press Enter.")

    # Load the model and data
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    data = UnstructuredCSVLoader('final.csv')
    index = VectorstoreIndexCreator().from_loaders([data])

    # Get user input
    question = st.text_input('Enter your query:', '')

    if st.button("Submit"):
        if question.strip() != '':
            # Get the response from the LangChain model
            response = index.query(llm=llm, question=question, chain_type="stuff")

            # Display the response
            st.markdown("### Response:")
            st.success(f"Result: {response}")

if __name__ == "__main__":
    main()

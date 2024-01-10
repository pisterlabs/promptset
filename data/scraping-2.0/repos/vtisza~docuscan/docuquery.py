import streamlit as st

from langchain.prompts.prompt import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

import docx2txt


class WordSuccessChecker:
    def __init__(self):
        self.uploaded_file = None
        self.doc_text = None
        self.input_text = None
        self._validation_prompt = """
        You are a highly trained assistant who reads through input documents and answer questions about them. Table elements are delimeted by | sign.
        This is the input document: {doc_text}

        You get the following question: {input_text}

        Provide a accurate and informative answer only based on the document. If question cannot be answered based on the input document do not make up an answer just tell answer is not in the text. """
        self.VALIDATION_PROMPT = PromptTemplate(
            input_variables=["doc_text", "input_text"],
            template=self._validation_prompt
        )

    def query_doc(self):
        if len(self.doc_text.split()) < 5000:
            return self.query_doc_short()
        else:
            return self.query_doc_long()


    def query_doc_short(self):
        llm_chain = LLMChain(
            llm=OpenAI(temperature=0, max_tokens=-1, model_name="gpt-3.5-turbo-16k"),
            prompt=self.VALIDATION_PROMPT
        )

        return llm_chain({"doc_text": self.doc_text, "input_text": self.input_text})["text"]
    
    def query_doc_long(self):
        llm_chain = LLMChain(
            llm=OpenAI(temperature=0, max_tokens=-1, model_name="gpt-3.5-turbo-16k"),
            prompt=self.VALIDATION_PROMPT
        )

        return llm_chain({"doc_text": self.doc_text[:14000], "input_text": self.input_text})["text"]

    def run(self):
        st.image("./logo1.png", width=150)
        st.title("AI Document Assistant")

        # Upload Word document
        self.uploaded_file = st.file_uploader("Upload a Word document", type=["docx"])

        if self.uploaded_file is not None:
            # Read the uploaded Word document
            self.doc_text = docx2txt.process(self.uploaded_file)

            # User input text
            self.input_text = st.text_input("Enter a question to query the uploaded document:")

            if st.button("Query"):
                if self.input_text:
                    result = self.query_doc()
                    st.write(f"Answer: {result}")
                else:
                    st.write("Please enter a question.")

if __name__ == '__main__':
    app = WordSuccessChecker()
    app.run()

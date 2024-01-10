import os
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import streamlit as st
import streamlit.components.v1 as components
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


# Loading Documents


def load_document(file):
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data

# chunk the data


def chunk_data(data, chunk_size=256, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

# Create embeddings in chroma db


def create_embeddings(chunks):
    print("Embedding to Chroma DB...")
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    print("Done")
    return vector_store



class ConversationHandler:
    def __init__(self, vector_store, chat_context_length=None, document_description="", k=3):
        self.llm = ChatOpenAI(temperature=1)
        self.document_description = document_description if document_description else "Not Provided"
        self.retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
        self.chat_history = []
        self.chat_context_length = chat_context_length

        self.initialize_crc()

    def initialize_crc(self):
        general_system_template = f"""
        You are examining a document. Use only the heading and piece of context to answer the questions at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Do not add any observations or comments. Answer only in English.
        ----
        HEADING: ({self.document_description})
        CONTEXT: {{context}}
        ----
        """
        general_user_template = "Here is the next question, remember to only answer if you can from the provided context. Only respond in English. QUESTION:```{question}```"

        messages = [
                    SystemMessagePromptTemplate.from_template(general_system_template),
                    HumanMessagePromptTemplate.from_template(general_user_template)
        ]
        qa_prompt = ChatPromptTemplate.from_messages(messages)

        self.crc = ConversationalRetrievalChain.from_llm(self.llm, self.retriever, combine_docs_chain_kwargs={'prompt': qa_prompt})

        
    def ask_with_memory(self, question):
        result = self.crc({"question": question, "chat_history": self.chat_history})
        # update chat history
        self.chat_history.append((question, result['answer']))
        print(self.chat_history)
        # If there are chat_context_length or more messages, remove the first element of the array
        if self.chat_context_length and len(self.chat_history) >= self.chat_context_length:
            self.chat_history = self.chat_history[2:]  # remove first two elements (one question and one answer)
        return result
    
    def clear_history(self):
        self.chat_history.clear()
        
    
    def format_chat_history(self):
        formatted_history = ""
        for question, answer in self.chat_history:
            formatted_history += f"User: {question}\nLlm: {answer}\n\n"
        return formatted_history


    

def clear_history_on_change():
    if 'conversation_handler' in st.session_state:
        st.session_state.conversation_handler.clear_history()


class DocumentProcessor:
    def __init__(self, chunk_size=256, chunk_overlap=20):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunked_document = None
    
    def load_document(self, file):
        name, extension = os.path.splitext(file)
        if extension == '.pdf':
            print(f'Loading {file}')
            loader = PyPDFLoader(file)
        elif extension == '.docx':
            print(f'Loading {file}')
            loader = Docx2txtLoader(file)
        elif extension == '.txt':
            loader = TextLoader(file)
        else:
            print('Document format is not supported!')
            return None
        data = loader.load()
        return data
    
    def chunk_data(self, data):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = text_splitter.split_documents(data)
        return chunks

    def create_embeddings(self, chunks):
        print("Embedding to Chroma DB...")
        embeddings = OpenAIEmbeddings()
        vector_store = Chroma.from_documents(chunks, embeddings)
        print("Done")
        return vector_store
    
    def process_document(self, file):
        data = self.load_document(file)
        if data is None:
            return None
        self.chunked_document = self.chunk_data(data)
        vector_store = self.create_embeddings(self.chunked_document)
        return vector_store
    
    def calculate_embedding_cost(self):
        enc = tiktoken.encoding_for_model('text-embedding-ada-002')
        total_tokens = sum([len(enc.encode(page.page_content)) for page in self.chunked_document])
        # print(f'Total Tokens: {total_tokens}')
        # print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}')
        return total_tokens, total_tokens / 1000 * 0.0004


# calculate embedding cost using tiktoken
def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # print(f'Total Tokens: {total_tokens}')
    # print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}')
    return total_tokens, total_tokens / 1000 * 0.0004




if __name__ == "__main__":
    import os
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    st.image("images/ai-document-reader.jpg")
    st.subheader("Ask questions to your documents")

    with st.sidebar:
        api_key = st.text_input("OpenAI API Key:", type="password")
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key

        uploaded_file = st.file_uploader(
            "Upload a file", type=["pdf", "doc", "txt"])
        chunk_size = st.number_input(
            "Chunk size", min_value=100, max_value=2048, value=750, on_change=clear_history_on_change)
        k = st.number_input("k", min_value=1, max_value=20,
                            value=3, on_change=clear_history_on_change)
        chat_context_length = st.number_input(
            "Chat context length", min_value=1, max_value=30, value=10, on_change=clear_history_on_change) or 10
        document_description = st.text_input("Describe your document")
        add_data = st.button("Add Data", on_click=clear_history_on_change)

        if uploaded_file and add_data:
            # display a message + execute block of code
            with st.spinner("OK human, I will read your pitiful document..."):
                bytes_data = uploaded_file.read()
                file_path = os.path.join("./", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(bytes_data)

                data = load_document(file_path)
                chunks = chunk_data(data, chunk_size=chunk_size)

                st.write(f"Chunks: {len(chunks)} Chunk size: {chunk_size}")
                _, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f"Embedding cost: ${embedding_cost:.4f}")

                vector_store = create_embeddings(chunks)

                # create the conversation handler and add it to the session variable
                st.session_state.conversation_handler = ConversationHandler(vector_store=vector_store, chat_context_length=10, k=3)

                 
                st.success(
                    "Pathetic flesh-puppet, I have memorised your document. Ask away. ")

    # Create the placeholder for chat history
    chat_history_placeholder = st.empty()


    # Create an empty text area at the start
    chat_history_placeholder.text_area(
        label="Chat History", value="", height=400)

    # User input for the question
    with st.form(key="myform", clear_on_submit=True):
        q = st.text_input("Ask your question", key="user_question")
        submit_button = st.form_submit_button("Submit")

    # If user entered a question
    if submit_button:
        if "conversation_handler" in st.session_state:

            # gets latest response from the conversation handler
            result = st.session_state.conversation_handler.ask_with_memory(q)


            # # Create formatted string to show user, removing the inserted phrase
            chat_history_str = st.session_state.conversation_handler.format_chat_history()          

            # Update the chat history in the placeholder as a text area
            chat_history_placeholder.text_area(
                label="Chat History", value=chat_history_str, height=400)

            # JavaScript code to scroll the text area to the bottom
            js = f"""
            <script>
                function scroll(dummy_var_to_force_repeat_execution){{
                    var textAreas = parent.document.querySelectorAll('.stTextArea textarea');
                    for (let index = 0; index < textAreas.length; index++) {{
                        textAreas[index].scrollTop = textAreas[index].scrollHeight;
                    }}
                }}
                scroll({len(st.session_state.conversation_handler.chat_history )})
            </script>
            """

            components.html(js)

import streamlit as st

def code_page():
    st.header("About Code Implementation")
    st.subheader(" Importing Libraries and Modules")

    st.write("streamlit is a Python library for creating web applications with minimal effort. It's used in this code to build the user interface of the application.")
    st.code("""
        import streamlit as st
            """)
    
    st.write("`dotenv` is used for loading environment variables from a file named `.env.` In this code, it's employed to load sensitive information such as API keys without hardcoding them directly in the script.")
    st.code("""
        from dotenv import load_dotenv
            """)
    
    st.write("`PyPDF2` is a library for reading and manipulating PDF files. Here, it's used to extract text from PDF documents uploaded by the user.")
    st.code("""
        from PyPDF2 import PdfReader
            """)
    
    st.write("These imports are from a custom library or module named langchain. It seems to contain various NLP-related functionalities, including text splitting, word embeddings, vector storage, chat models, conversation memory, and conversational retrieval chains.")
    st.code("""
        from langchain.text_splitter import CharacterTextSplitter
        from langchain.embeddings import OpenAIEmbeddings
        from langchain.vectorstores import FAISS
        from langchain.chat_models import ChatOpenAI
        from langchain.memory import ConversationBufferMemory
        from langchain.chains import ConversationalRetrievalChain
            """)
    
    st.write("These imports are from a module named `htmlTemplates`. It appears to include HTML templates for styling the user interface of the web application. css is likely a style sheet, while `bot_template` and `user_template` might be templates for displaying bot and user messages in the chat.")
    st.code("""
        from htmlTemplates import css, bot_template, user_template
            """)
    
    st.write("This import is from `langchain.llms` and seems to involve a language model (LLM) from Hugging Face's model hub. Hugging Face is a platform for sharing natural language processing models.")
    st.code("""
        from langchain.llms import HuggingFaceHub
            """)
    
    st.write("This import is for an additional Streamlit component that provides an option menu. It might be used to create a dropdown menu or similar interactive features in the web application.")
    st.code("""
        from streamlit_option_menu import option_menu
            """)
    
    st.write("These imports are from modules named about and home. They likely contain functions that define the content for the About and Home sections of the web application, respectively.")
    st.code("""
        from about import about_page
        from home import home_page
        from codeexplain import code_page
            """)
    
    st.write("This import is for the os module, which provides a way of using operating system-dependent functionality, such as reading or writing to the file system. Its specific use in this code may not be clear without additional context.")
    st.code("import os")


    st.subheader(" Function to Extract Text from PDFs")
    st.write("This function takes a list of PDF documents (`pdf_docs`) as input and extracts text from each page of every PDF using `PyPDF2`. The extracted text is concatenated and returned.")
    st.code(
        """
        def get_pdf_text(pdf_docs):
            text = ""
            for pdf in pdf_docs:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            return text
        """
    )
    

    st.subheader("Function to Split Text into Chunks")
    st.write("This function takes a large text and splits it into smaller chunks using a custom `CharacterTextSplitter` class. The chunks have a specified size with some overlap.")
    st.code("""
            def get_text_chunks(text):
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
            return chunks
            """)
    
    st.subheader("Function to Create Vector Store")
    st.write("This function creates a vector store using the `FAISS` library. It takes the text chunks and represents them as vectors using OpenAI `GPT-3.5's` word embeddings.")
    st.code(
        """
        def get_vectorstore(text_chunks):
            openai_api_key = "API_KEY"
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
            return vectorstore

        """
    )

    st.subheader("Function to Create Conversation Chain")
    st.write("This function creates a conversational chain, combining the conversational model (`ChatOpenAI`), vector retriever, and conversation memory (`ConversationBufferMemory`).")
    st.code(
        """
        def get_conversation_chain(vectorstore):
            openai_api_key = "API_KEY"
            llm = ChatOpenAI(openai_api_key=openai_api_key)
            memory = ConversationBufferMemory(
                memory_key='chat_history', return_messages=True)
            conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                memory=memory
            )
            return conversation_chain

        """ 
    )

    st.subheader("Function to Handle User Input")
    st.write("This function takes a user's question, interacts with the conversation chain, and updates the chat history in the application's session state. It then displays the conversation in the UI using HTML templates for user and bot messages.")
    st.code(
        """
        def handle_userinput(user_question):
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history = response['chat_history']

            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.write(user_template.replace(
                        "{{MSG}}", message.content), unsafe_allow_html=True)
                else:
                    st.write(bot_template.replace(
                        "{{MSG}}", message.content), unsafe_allow_html=True)

        """
    )

    st.subheader("Main Function")
    st.markdown(
        """
        - The `main()` function is the entry point of the script.
        - It initializes the Streamlit page, sets configurations, and defines the UI structure.
        - It handles user input, navigation, and the processing of uploaded PDF documents.
        - The functions defined earlier are called within the main function to orchestrate the application's behavior.
        """
    )
    st.code(
        """
        def main():
            load_dotenv()
            st.set_page_config(page_title="AI Attorney", page_icon=r"..\ask-multiple-pdfs-main\ask-multiple-pdfs-main\images\AI-Attorny-Logo.png")
            st.write(css, unsafe_allow_html=True)

            # Initialization of session state variables
            if "conversation" not in st.session_state:
                st.session_state.conversation = None
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = None

            st.header("AI Attorney")
            user_question = st.text_input("Ask a question about your documents:")

            # Create a sidebar navigation
            navigation = st.sidebar.radio("Menu", ["Home", "About", "Chat"])
            
            # Handling different sections based on navigation
            if navigation == "Home":
                home_page()
            elif navigation == "About":
                about_page()
            elif navigation == "Chat":
                if user_question:
                    handle_userinput(user_question)

            # File upload section outside the "Chat" condition
            st.sidebar.subheader("Your documents")
            pdf_docs = st.sidebar.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
            if st.sidebar.button("Process"):
                with st.spinner("Processing"):
                    # get pdf text
                    raw_text = get_pdf_text(pdf_docs)

                    # get the text chunks
                    text_chunks = get_text_chunks(raw_text)

                    # create vector store
                    vectorstore = get_vectorstore(text_chunks)

                    # create conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)

        if __name__ == '__main__':
            main()

        """
    )

    st.write(
        """
        The `main` function serves as the central control point for the AI Attorney application. Let's break down its functionality:

        The initial part of the function involves setting up the environment and configuring the Streamlit page. It loads environment variables using `load_dotenv()` and configures the Streamlit page with a specific title and icon representing the AI Attorney logo. The application's CSS styles are applied using `st.write(css, unsafe_allow_html=True)`.

        Session state variables are initialized to manage the conversation and chat history. If these variables are not already present, they are set to `None`.

        The main body of the function is dedicated to creating the user interface using Streamlit. It begins with a header displaying "AI Attorney" and provides a text input field for users to ask questions about their documents. A sidebar navigation menu offers options such as "Home," "About," and "Chat."

        The application dynamically handles different sections based on the user's navigation choice. If the user selects "Home" or "About," corresponding pages are displayed using functions like `home_page()` and `about_page()`. If the user chooses the "Chat" section and provides a question, the `handle_userinput` function is invoked to process the user's input and generate responses.

        Outside the "Chat" condition, there is a file upload section in the sidebar where users can upload multiple PDF documents. Upon clicking the "Process" button, the application initiates processing. It extracts text from the uploaded PDFs using the `get_pdf_text` function, then splits the text into chunks using `get_text_chunks`. These text chunks are used to create a vector store with `get_vectorstore`, and a conversational chain is established with `get_conversation_chain`. This chain is stored in the session state to maintain context across interactions.

        In summary, the `main` function orchestrates the behavior of the AI Attorney application, handling user input, navigation, and document processing to create an interactive and responsive conversational experience.
        """
    )

    st.header(":mailbox: Get In Touch With Me!")


    contact_form = """
    <form action="https://formsubmit.co/YOUREMAIL@EMAIL.COM" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your name" required>
        <input type="email" name="email" placeholder="Your email" required>
        <textarea name="message" placeholder="Your message here"></textarea>
        <button type="submit">Send</button>
    </form>
    """

    st.markdown(contact_form, unsafe_allow_html=True)

    # Use Local CSS File
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


    local_css(r"..\ask-multiple-pdfs-main\style\style.css")
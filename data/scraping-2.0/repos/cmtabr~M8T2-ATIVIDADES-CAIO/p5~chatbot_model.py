# Ponderada 5

# Imports
import gradio as gr
from langchain.llms import ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import  RunnablePassthrough
from langchain.document_loaders import SeleniumURLLoader, DirectoryLoader, TextLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import chroma
from langchain.chains import RetrievalQA

model = ollama.Ollama(model="llama2")

# urls = ["https://www.deakin.edu.au/students/study-support/faculties/sebe/abe/workshop/rules-safety"]

def archive_loader_and_vectorizer():
        """ 
        This function loads txt documents from current directory 
        and vectorizes them
        """
        # loader = SeleniumURLLoader(urls=urls)
        loader = DirectoryLoader('../', 
                                glob='**/*.txt',
                                loader_cls=TextLoader,
                                show_progress=True
                            )

        print("Loading data...")

        data = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=30000, chunk_overlap=0)

        docs = text_splitter.split_documents(data)

        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        vectorstore = chroma.Chroma.from_documents(docs, embedding_function)

        retriever = RetrievalQA.from_chain_type(llm=ollama.Ollama(), chain_type="stuff", retriever=vectorstore.as_retriever(search_kwargs={"k": 1}))

        return retriever


def chatbot_manager(text):
    template = """
    From now on, You will be my personal security manager. And when i ask you for advice, 
    using the following {context} you will give me the best advice.
    {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    retriever = archive_loader_and_vectorizer()
    chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | model
            )
    
    complete_output = []

    try:
        print(f"Sua solicitação foi realizada: {text}")
        msg = ""
        for s in chain.stream(text):
            print(s, end="", flush=True)
            msg += s
            yield msg
        complete_output.append((text, msg))
        return "", complete_output
    
    except Exception as e:
        print(f"Erro ao enviar solicitação para OLLAMA: {e}")
        return "Erro ao enviar solicitação para OLLAMA", complete_output

screen = gr.Interface(
    fn=chatbot_manager,
    title="Welcome to Your Personal Security Assistant!",
    description="Ask for an advice and ensure your safety!",
    inputs="text",
    outputs="text",
    examples=["Who is allowed to operate a lathe? What protective gear should be used to do it?",
            "What are the risks of operating a lathe? What are the safety measures to be taken?",
            "What kinda of rules must be followed when using machinery?"
    ],
    theme=gr.themes.Soft()
).queue()

screen.launch()

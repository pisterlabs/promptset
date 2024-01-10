from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
import gradio as gr
import re
from chatbot_actions import chatbot_print
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings


intent_dict = {
    r"Indo pegar [ao]?\s*([\w\s]+)": chatbot_print
}

def send_request(user_prompt):
    # load the document and split it into chunks
    load_dotenv()

    model = ChatOpenAI(model="gpt-3.5-turbo")

    loader = PyPDFLoader("./data/points.pdf")
    pages = loader.load_and_split()

    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )

    vectorstore = FAISS.from_documents(pages, OpenAIEmbeddings())

    retriever = vectorstore.as_retriever()

    print(retriever)

    template = """Responda a pergunta a seguir com a seguinte lista de ferramentas de contexto:
    {context}

    Caso seja pedido para você pegar uma ferramenta, responda apenas e somente com: "Indo pegar o/a [ferramenta]"

    Pergunta: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOpenAI(model="gpt-3.5-turbo")

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
    )

    print("\nModel answer:\n")
    last_string = ""
    for s in chain.stream(user_prompt):
        last_string += s.content
    print(last_string)
    return last_string

def generate_response(prompt, chat_history):
    response = send_request(prompt)
    chat_history.append((prompt, response))
    for key, function in intent_dict.items():
        pattern = re.compile(key)
        point = pattern.findall(response)
        if point:
            function(point[0])
    
    return "", chat_history

with gr.Blocks() as demo:
    title="LLM Chatbot"
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    msg.submit(generate_response, [msg, chatbot], [msg, chatbot])

# Execute a interface Gradio
if __name__ == "__main__":
    demo.launch()

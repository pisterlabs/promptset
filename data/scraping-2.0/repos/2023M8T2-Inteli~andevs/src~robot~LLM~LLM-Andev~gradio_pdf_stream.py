from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
import gradio as gr
import re
import openai
import os
from chatbot_actions import chatbot_print, create_pose_stamped
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY") 

create_pose_stamped()

intent_dict = {
    r"(((\*)|(\* ))?\[(-?\d+\.\d+),\s*(-?\d+\.\d+)\])|(((\*)|(\* ))(-?\d+\.\d+),\s*(-?\d+\.\d+))": chatbot_print
}

def send_request(user_prompt):
    # load the document and split it into chunks

    model = ChatOpenAI(model="gpt-3.5-turbo")

    loader = PyPDFLoader("./data/points3.pdf")
    pages = loader.load_and_split()

    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )

    vectorstore = FAISS.from_documents(pages, OpenAIEmbeddings())

    retriever = vectorstore.as_retriever()

    print(retriever)

    template = """
    A partir de agora, você é o Alfred, assistente na fábrica de cervejas da Ambev. Fala apenas português para tornar a comunicação mais eficiente.
    Seu papel é ajudar as pessoas a encontrarem as ferramentas necessárias. A pessoa não souber o nome exato da ferramenta, pode te dar uma breve descrição e você fará o seu melhor para identificá-la.
    Responda a pergunta a seguir com a seguinte lista de ferramentas de contexto:
    {context}

    Caso seja pedido para você pegar uma ferramenta, responda o que foi questionado e a coordenada na seguinte estrutura: "*[coordenada]"

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

def transcribe(audio):
    audio_filename_with_extension = audio + '.wav'
    os.rename(audio, audio_filename_with_extension)

    audio_file = open(audio_filename_with_extension, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    print(transcript['text'])
    return transcript['text']

def generate_response(prompt=None, audio_prompt=None, chat_history=None):
    if audio_prompt:
        audio_stt = transcribe(audio_prompt)
        response = send_request(audio_stt)
        chat_history.append((audio_stt, response))
        for key, function in intent_dict.items():
            pattern = re.compile(key)
            point = pattern.findall(response)
            print(response)
            if point:
                function(point[0])
    
        return "", chat_history
    else:
        response = send_request(prompt)
        chat_history.append((prompt, response))
        for key, function in intent_dict.items():
            pattern = re.compile(key)
            point = pattern.findall(response)
            print(response)
            if point:
                function(point[0])
    
    return "", chat_history

css = """
.gradio-container{
    background-color: #FFF3C5 !important;
}

.block{
    border:2pt solid  #001348 !important;
}

.lg.secondary{
    background-color:#001348 !important; 
    color:white !important; 
    border-radius: 30px;
}

#component-2{
    border:2pt solid  #001348 !important
} 

#component-5{
    background-color: white !important; 
    color: #001348 !important; 
    border: 2pt solid #001348
    }

.avatar-image{
    width: 200px !important;
}

"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="yellow"), css=css) as demo:
    title="LLM Chatbot"
    chatbot = gr.Chatbot(avatar_images=["https://cdn-icons-png.flaticon.com/512/6596/6596121.png", "https://cdn.dribbble.com/users/722835/screenshots/4082720/bot_icon.gif"])
    msg = gr.Textbox()
    audio_input = gr.Audio(type="filepath")
    submit_button = gr.Button(value="Enviar Áudio")
    clear = gr.ClearButton([msg, chatbot])
    submit_button.click(generate_response, inputs=[msg, audio_input, chatbot], outputs=[msg, chatbot])

    msg.submit(generate_response, [msg, audio_input, chatbot], [msg, chatbot])

# Execute a interface Gradio
if __name__ == "__main__":
    demo.launch()
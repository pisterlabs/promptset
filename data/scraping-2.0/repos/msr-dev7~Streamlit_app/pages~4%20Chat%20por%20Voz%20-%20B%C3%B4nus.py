import streamlit as st
import speech_recognition as sr
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from gtts import gTTS
import os
openai_api_key = st.secrets["my_secret"]["OPENAI_API_KEY"]

if openai_api_key:
    chat = ChatOpenAI(api_key=openai_api_key)

    llm = ConversationChain(llm=chat)

    def recognize_speech():
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.write("Fale algo...")
            audio = recognizer.listen(source)
            try:
                text = recognizer.recognize_google(audio, language='pt-BR')  # STT para texto
                return text
            except sr.UnknownValueError:
                return "Desculpe, não entendi o áudio."
            except sr.RequestError:
                return "Não foi possível acessar o serviço de reconhecimento de fala."

    # Função para interagir com o ChatGPT
    def chat_with_gpt(prompt):
        try:
            saida = llm.run(prompt)
            return saida
        except Exception as e:
            return f"Erro ao interagir com o modelo: {str(e)}"

    # Função para síntese de voz
    def text_to_speech(text):
        tts = gTTS(text=text, lang='pt-br')
        tts.save("response.mp3")
        os.system("mpg321 response.mp3")



    # Interface do Streamlit
    st.header("Chat por Voz")
    st.markdown("""
    Nesta versão não conseguimos habilitar o microfone e por isso dá erro, segue **[Código do Chat por Voz com microfone local](https://drive.google.com/drive/folders/1GnYlavzy7NBetLSA71Sx0xzYHEb1kEWM)** 
    
    Em outra tentativa fizemos com streamlit_webrtc, conseguimos habilitar o microfone do navegador, mas a usabilidade não ficou legal, **[ver versão publicado aqui](https://chatvoz-3weauazvbnktgqiceje6r9.streamlit.app/)**""")
    c0,_,c1 = st.columns(3)
    
    with c0:
        inicio = st.button("Iniciar Interacao por Voz")

    with c1:
        stop_button = st.button("Encerrar Conversa", key="encerrar_conversa")

    
    if "mensagens" not in st.session_state:
        st.session_state.mensagens = [{"role": 'system', "content": 'Você será um amigo para conversar qualquer coisa.'}] 

    interagindo = False
    # Restante do seu código
    
    if inicio:
        interagindo = True


    while interagindo:
        input_text = recognize_speech()
        if input_text and "Desculpe, não entendi o áudio." not in input_text:
            #conversa.append(("Usuário:", input_text))  # Adiciona entrada do usuário ao histórico
            st.session_state.mensagens.append({"role": 'user', "content": input_text})
            #st.write(f"Transcrição do áudio: {input_text}")
            with st.chat_message("user"):
                    st.markdown(input_text)
            response_text = chat_with_gpt(input_text)
            #conversa.append(("Assistente:", response_text))  # Adiciona resposta do modelo ao histórico
            st.session_state.mensagens.append({"role": 'system', "content": response_text})
            #st.write(f"Assistente: {response_text}")
            with st.chat_message("system"):
                    st.markdown(response_text)
            if "Erro ao interagir com o modelo" not in response_text:
                text_to_speech(response_text)
                st.audio("response.mp3")
        else:
            st.write("Não foi detectado texto na fala. Por favor, fale novamente.")

        if stop_button:
            interagindo = False  # Encerra a interação


    if st.session_state.mensagens != [{"role": 'system', "content": 'Você será um amigo para conversar qualquer coisa.'}]:            
            st.markdown("Histórico da Conversa:")
    
    for mensagens in st.session_state.mensagens[1:]:
        with st.chat_message(mensagens["role"]):
            st.markdown(mensagens["content"])
    
    st.session_state.mensagens = [{"role": 'system', "content": 'Você será um amigo para conversar qualquer coisa.'}]             
    

#import libraries
import streamlit as st
import replicate
import os
from langchain.llms import Replicate
from langchain import PromptTemplate, LLMChain

# App title
st.set_page_config(page_title="Llama2🦙🤖 ")

#function
def generate_response(txt):
    # Instantiate the Llama2 model
    llm = Replicate(model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
                    input={"temperature":0.03, "top_p":0.8, "max_length":900})

    template = """
        Escribe un resumen conciso del texto, devuelve tus respuestas con 5 líneas que cubran los puntos clave del texto..
        ```{text}```
        RESUMEN:
        """
    prompt = PromptTemplate(template=template, input_variables=["text"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    summary = llm_chain.run(txt)
    return summary


#new
page = st.sidebar.selectbox("Seleccione una función", ("Chat", "Resumen"))
# verify key
with st.sidebar:
    if 'REPLICATE_API_TOKEN' in st.secrets:
        st.success('API key already provided!', icon='✅')
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
    os.environ['REPLICATE_API_TOKEN'] = replicate_api


if page == "Chat":
    #credentials
    with st.sidebar:
        st.title('Llama2 Chatbot🦙')
        
    
        st.subheader('Models and parameters')
        selected_model = st.sidebar.selectbox('Escoja el modelo Llama2', ['Llama2-7B', 'Llama2-13B'], key='selected_model')
        if selected_model == 'Llama2-7B':
            llm = 'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea'
        elif selected_model == 'Llama2-13B':
            llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
        temperature = st.sidebar.slider('creatividad del modelo', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
        top_p = st.sidebar.slider('creatividad de palabras', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
        max_length = st.sidebar.slider('longitud de respuestas', min_value=32, max_value=600, value=420, step=8)
    
    
    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "¿En qué puedo ayudarte?"}]
    
    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "¿En qué puedo ayudarte?"}]
    st.sidebar.button('Limpiar Chat', on_click=clear_chat_history)
    
    # Function for generating LLaMA2 response. Refactored from https://github.com/a16z-infra/llama2-chatbot
    def generate_llama2_response(prompt_input):
        string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
        for dict_message in st.session_state.messages:
            if dict_message["role"] == "user":
                string_dialogue += "User: " + dict_message["content"] + "\n\n"
            else:
                string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
        output = replicate.run('a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5', 
                               input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
                                      "temperature":temperature, "top_p":top_p, "max_length":max_length, "repetition_penalty":1})
        return output
    
    # User-provided prompt
    if prompt := st.chat_input(disabled=not replicate_api):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
    
    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_llama2_response(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)

    
elif page == "Resumen":
    with st.sidebar:
        st.title('Resumen con Llama2🦙')

    # Text input
    txt_input = st.text_area('Introduzca el texto', '', height=200)

    # Form to accept user's text input for summarization
    result = []

    submitted = st.button('Resumir')
    if submitted:
        with st.spinner('Resumiendo...'):
            response = generate_response(txt_input)
            result.append(response)

    if len(result):
        st.info(result[0])

    


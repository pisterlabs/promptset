import streamlit as st
import logging
import os
import time

from src.main import Memory, Embeddings, VectorStore, Chain, Weights, Template, LLM

from langchain.prompts import PromptTemplate

logging.basicConfig(level=logging.DEBUG)

def callback():
    st.session_state.lock_widget = True

def main():
    st.title("JuridIA")
    
    source_documents = None
    
    if 'lock_widget' not in st.session_state:
        st.session_state.lock_widget = False
    
    if 'db' not in st.session_state:
        st.session_state.db = None
        
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "sources" not in st.session_state:
        st.session_state.sources = []
    
    if "query" not in st.session_state:
        st.session_state.query = False
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    tab1, tab2, tab3, tab4, tab5 = st.tabs(['Consulta', 'Fuentes', 'Documentos', 'Configuración', 'Depurar'])
    
    with tab1:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    with tab3:
        
        files = st.file_uploader("Subir archivo", type=["pdf"], accept_multiple_files=True, key='filetab3', disabled=st.session_state.lock_widget)
        
        with st.expander(label="Configuración", expanded=False):
            emb_options = [key for dictionary in Embeddings.get_dict() for key in dictionary.keys()]
            repo_id = st.selectbox(label='Modelo de Embeddings', options=emb_options, index=0, disabled=st.session_state.lock_widget)
            chunk_size = st.slider(label='Tamaño de texto', min_value=128, max_value=512, step=8, value=512, disabled=st.session_state.lock_widget)
            chunk_overlap = st.slider(label='Tamaño de sobrelape', min_value=0, max_value=256, step=8, value=0, disabled=st.session_state.lock_widget)
            name = st.text_input(label='Nombre de indice', value='ejemplo', disabled=st.session_state.lock_widget)
            checkbox = st.checkbox(label='Usar GPU?', disabled=st.session_state.lock_widget, help='habilita el uso de la GPU para la creacion de los indices')
            
            if checkbox:
                device = 'cuda'
            else:
                device = 'cpu'
                
        if st.button(label='Procesar Documentos', key='btntab3', disabled=st.session_state.lock_widget, on_click=callback):
            with st.spinner('Por favor espere...'):
                os.makedirs('./Documents', exist_ok=True)
                for file in files:
                    file_path = os.path.join('Documents', file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                try:
                    st.session_state.db = VectorStore.get(name=name, chunk_size=chunk_size, chunk_overlap=chunk_overlap, model_name=repo_id, device=device)
                except ValueError:
                    st.write('No hay documentos presentes')
                except AssertionError:
                    st.write('CUDA no esta disponible (usar conda)')
                         
                time.sleep(1)
            st.session_state.lock_widget = False
            st.experimental_rerun()
    
    with tab4:
        k = st.slider(label='Numero de documentos a buscar', min_value=1, max_value=10, step=1, value=5, disabled=st.session_state.lock_widget)
        with st.expander(label="Configuración de modelo de lenguaje", expanded=False):
            
            top_p=st.slider(label='Top P', min_value=0.1, max_value=1.0, step=0.1, value=0.1, key='top_p_webui', disabled=st.session_state.lock_widget, help='If not set to 1, select tokens with probabilities adding up to less than this number. Higher value = higher range of possible random results.')
            top_k=st.slider(label='Top K', min_value=1, max_value=100, step=1, value=40, key='top_k_webui', disabled=st.session_state.lock_widget, help='Similar to top_p, but select instead only the top_k most likely tokens. Higher value = higher range of possible random results.')
            temperature=st.slider(label='Temperatura', min_value=0.0, max_value=2.0, step=0.1, value=0.1,key='temp_webui', disabled=st.session_state.lock_widget, help='Primary factor to control randomness of outputs. 0 = deterministic (only the most likely token is used). Higher value = more randomness.') 
            max_new_tokens = st.number_input(label='Tokens nuevos maximos', min_value=0, value=2048,key='tokens_webui', disabled=st.session_state.lock_widget, help='max number of new tokens to generate')
            
            llm_type = st.selectbox(label='Seleccionar Backend de Modelo de lenguaje', options=['TextGen', 'OpenAI', 'LlamaCpp'], index=0, disabled=st.session_state.lock_widget)
            if llm_type=='TextGen':
                local_Host = st.text_input(label='API URL', value='http://127.0.0.1:5000', disabled=st.session_state.lock_widget)
                llm = LLM.get(llm=llm_type, top_p=top_p, top_k=top_k, temperature=temperature, max_tokens=max_new_tokens, local_Host=local_Host)
            elif llm_type=='OpenAI':
                OpenAi_Host = st.text_input(label='API URL', value='https://api.openai.com/v1', disabled=st.session_state.lock_widget)
                OpenAi_Key = st.text_input(label='TU KEY DE OPEN AI', value='sk-111111111111111111111111111111111111111111111111', disabled=st.session_state.lock_widget)
                OpenAi_Model = st.selectbox(label='MODELO DE OPEN AI', options=['text-davinci-003', 'gpt-3.5-turbo-0301'], index=0, disabled=st.session_state.lock_widget)
                try:
                    llm = LLM.get(llm=llm_type, top_p=top_p, top_k=top_k, temperature=temperature, max_tokens=max_new_tokens, OpenAi_Host=OpenAi_Host, OpenAi_Key=OpenAi_Key, OpenAi_Model=OpenAi_Model)
                except ImportError:
                    st.write('Could not import openai python package. Please install it with `pip install openai`')
            else:
                llm_options = [value for dictionary in Weights.get_dict() for value in dictionary.values()]
                local_Model = st.selectbox(label='Seleccionar Modelo local a descargar', options=llm_options, index=2, disabled=st.session_state.lock_widget)
                if st.button(label='Descargar Modelo', key='download', disabled=st.session_state.lock_widget, on_click=callback):
                    with st.spinner('Por favor espere...'):
                        Weights.get(model=local_Model)
                    st.session_state.lock_widget = False
                    st.experimental_rerun()
                try:
                    llm = LLM.get(llm=llm_type, top_p=top_p, top_k=top_k, temperature=temperature, max_tokens=max_new_tokens, model_path=f'./Models/{local_Model}')
                except ImportError:
                    st.write('Could not import llama.cpp python package. Please install it with `pip install llama-cpp-python`')
           
        with st.expander(label="Plantilla de pregunta condensada", expanded=False):
            system_name = st.text_input(label='nombre del mensaje del sistema', value='### System:', disabled=st.session_state.lock_widget)
            system_text = st.text_area(label='texto del mensaje del sistema', value='Eres un asistente de IA que sigue instrucciones extremadamente bien. Ayuda tanto como puedas.', disabled=st.session_state.lock_widget)
            user_name = st.text_input(label='nombre del mensaje del usuario', value='### User:', disabled=st.session_state.lock_widget)
            user_text = st.text_area(label='texto del mensaje del usuario', value='Dada la siguiente conversación y una pregunta de seguimiento, reformula la pregunta de seguimiento para que sea una pregunta independiente, en su idioma original.', disabled=st.session_state.lock_widget)
            input_name = st.text_input(label='nombre del mensaje de entrada', value='### Input:', disabled=st.session_state.lock_widget)
            input_text = st.text_area(label='texto del mensaje de entrada', value='Historial del chat:\n{chat_history}\nPregunta de seguimiento: {question}', disabled=st.session_state.lock_widget)
            response_name = st.text_input(label='nombre del mensaje de respuesta', value='### Response:', disabled=st.session_state.lock_widget)
            response_text = st.text_area(label='texto del mensaje de respuesta', value='Pregunta independiente:', disabled=st.session_state.lock_widget)
            Condense_template = Template.get(sys_name=system_name, system=system_text, user_name=user_name, user=user_text, input_name=input_name, input=input_text, res_name=response_name, response=response_text)
            Condense_template_preview = st.code(body=Condense_template)

        with st.expander(label="Plantilla de pregunta y respuesta", expanded=False):
            system_name = st.text_input(label='nombre del mensaje del sistema', value='### System:', key='qa_sys_name', disabled=st.session_state.lock_widget)
            system_text = st.text_area(label='texto del mensaje del sistema', key='qa_sys', value='Eres un asistente de IA que sigue instrucciones extremadamente bien. Ayuda tanto como puedas.', disabled=st.session_state.lock_widget)
            user_name = st.text_input(label='nombre del mensaje del usuario', value='### User:', key='qa_user_name', disabled=st.session_state.lock_widget)
            user_text = st.text_area(label='texto del mensaje del usuario', key='qa_user', value='''Utiliza los siguientes fragmentos de contexto para responder la pregunta al final. Si no sabes la respuesta, simplemente di que no lo sabes, no intentes inventar una respuesta.''', disabled=st.session_state.lock_widget)
            input_name = st.text_input(label='nombre del mensaje de entrada', value='### Input:', key='qa_input_name', disabled=st.session_state.lock_widget)
            input_text = st.text_area(label='texto del mensaje de entrada', key='qa_input', value='Pregunta: {question}\nFragmentos de contexto:\n{context}', disabled=st.session_state.lock_widget)
            response_name = st.text_input(label='nombre del mensaje de respuesta', value='### Response:', key='qa_response_name', disabled=st.session_state.lock_widget)
            response_text = st.text_area(label='texto del mensaje de respuesta', value='Respuesta útil:', key='qa_response', disabled=st.session_state.lock_widget)
            QA_template = Template.get(sys_name=system_name, system=system_text, user_name=user_name, user=user_text, input_name=input_name, input=input_text, res_name=response_name, response=response_text)
            QA_template_preview = st.code(body=QA_template)


    if prompt := st.chat_input(placeholder="Consultar", key='user_input', disabled=st.session_state.lock_widget, on_submit=callback):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.spinner('Generando Respuesta'):
            if st.session_state.db is None:
                response = 'No hay documentos presentes, suba algunos documentos'
                st.session_state.messages.append({"role": "assistant", "content": response})
                time.sleep(1)
            else:
                db = st.session_state.db
                retriever = VectorStore.retriever(db=db, k=k)
                chain = Chain.get_no_mem(llm=llm, retriever=retriever, condense_question_prompt=PromptTemplate.from_template(template=Condense_template),
                                  qa_prompt=PromptTemplate.from_template(template=QA_template))
                try:
                    response, source_documents  = Chain.query_no_mem(prompt=prompt, chain=chain, chat_history=st.session_state.chat_history)
                    history = (prompt, response)
                    st.session_state.chat_history.append(history)
                except:
                    response = 'cant stablish connection to llm backend'
                st.session_state.sources = source_documents
                st.session_state.query = True
                st.session_state.messages.append({"role": "assistant", "content": response})
                time.sleep(1)
        st.session_state.lock_widget = False
        st.experimental_rerun()
    
    with tab2:
        if st.session_state.query:
            sources = st.session_state.sources
            if sources is not None:
                for source in sources:
                    source_name = os.path.basename(source)
                    st.markdown(f'##### Fuente: {source_name}\n')
                    source_data = sources[source]
                    sorted_data = sorted(source_data.items(), key=lambda x: int(x[0]))
                    
                    for page, content in sorted_data:
                        with st.expander(label=f'Pagina: {page}', expanded=False):
                            st.markdown(content)
        else:
            sources = st.markdown(body='Pregunte algo primero')
    
    with tab5:
        with st.expander(label='chat_history'):
            st.write(st.session_state['chat_history'])
        with st.expander(label='db'):
            st.write(st.session_state['db'])
                        
if __name__ == '__main__':
    main()
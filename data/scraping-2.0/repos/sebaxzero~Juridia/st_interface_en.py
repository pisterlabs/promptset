import streamlit as st
import logging
import os
import time

from src.main import Memory, Embeddings, VectorStore, Chain, Weights, Template, LLM

from langchain.prompts import PromptTemplate

logging.basicConfig(level=logging.CRITICAL)

def callback():
    st.session_state.lock_widget = True

def main():
    st.title("ChainPDF")
    
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
        
    tab1, tab2, tab3, tab4, tab5 = st.tabs(['Query', 'Sources', 'Documents', 'Settings', 'debug'])
    
    with tab1:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    with tab3:
        
        files = st.file_uploader("Upload file", type=["pdf"], accept_multiple_files=True, key='filetab3', disabled=st.session_state.lock_widget)
        
        with st.expander(label="settings", expanded=False):
            emb_options = [key for dictionary in Embeddings.get_dict() for key in dictionary.keys()]
            repo_id = st.selectbox(label='Embeddings model', options=emb_options, index=0, disabled=st.session_state.lock_widget)
            chunk_size = st.slider(label='Chunk size', min_value=128, max_value=512, step=8, value=512, disabled=st.session_state.lock_widget)
            chunk_overlap = st.slider(label='Chunk overlap', min_value=0, max_value=256, step=8, value=0, disabled=st.session_state.lock_widget)
            name = st.text_input(label='index name', value='example', disabled=st.session_state.lock_widget)
            checkbox = st.checkbox(label='use gpu', disabled=st.session_state.lock_widget, help='enable the use of gpu for the vectorstore creation')
            
            if checkbox:
                device = 'cuda'
            else:
                device = 'cpu'
                
        if st.button(label='Process Documents', key='btntab3', disabled=st.session_state.lock_widget, on_click=callback):
            with st.spinner('please wait, this will take some time...'):
                os.makedirs('./Documents', exist_ok=True)
                for file in files:
                    file_path = os.path.join('Documents', file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                try:
                    st.session_state.db = VectorStore.get(name=name, chunk_size=chunk_size, chunk_overlap=chunk_overlap, model_name=repo_id, device=device)
                except ValueError:
                    st.write('No documents present at "Documents" folder')
                except AssertionError:
                    st.write('CUDA is not available')
                         
                time.sleep(1)
            st.session_state.lock_widget = False
            st.experimental_rerun()
    
    with tab4:
        k = st.slider(label='number of relevant_documents', min_value=1, max_value=10, step=1, value=5, disabled=st.session_state.lock_widget)
        with st.expander(label="llm settings", expanded=False):
            
            top_p=st.slider(label='Top P', min_value=0.1, max_value=1.0, step=0.1, value=0.1, key='top_p_webui', disabled=st.session_state.lock_widget, help='If not set to 1, select tokens with probabilities adding up to less than this number. Higher value = higher range of possible random results.')
            top_k=st.slider(label='Top K', min_value=1, max_value=100, step=1, value=40, key='top_k_webui', disabled=st.session_state.lock_widget, help='Similar to top_p, but select instead only the top_k most likely tokens. Higher value = higher range of possible random results.')
            temperature=st.slider(label='Temperature', min_value=0.0, max_value=2.0, step=0.1, value=0.1,key='temp_webui', disabled=st.session_state.lock_widget, help='Primary factor to control randomness of outputs. 0 = deterministic (only the most likely token is used). Higher value = more randomness.') 
            max_new_tokens = st.number_input(label='max new tokens', min_value=0, value=2048,key='tokens_webui', disabled=st.session_state.lock_widget, help='max number of new tokens to generate')
            
            llm_type = st.selectbox(label='select llm API', options=['TextGen', 'OpenAI', 'LlamaCpp'], index=0, disabled=st.session_state.lock_widget)
            if llm_type=='TextGen':
                local_Host = st.text_input(label='api url', value='http://127.0.0.1:5000', disabled=st.session_state.lock_widget)
                llm = LLM.get(llm=llm_type, top_p=top_p, top_k=top_k, temperature=temperature, max_tokens=max_new_tokens, local_Host=local_Host)
            elif llm_type=='OpenAI':
                OpenAi_Host = st.text_input(label='api url', value='https://api.openai.com/v1', disabled=st.session_state.lock_widget)
                OpenAi_Key = st.text_input(label='your OpenAI Key', value='sk-111111111111111111111111111111111111111111111111', disabled=st.session_state.lock_widget)
                OpenAi_Model = st.selectbox(label='select OpenAI Model', options=['text-davinci-003', 'gpt-3.5-turbo-0301'], index=0, disabled=st.session_state.lock_widget)
                try:
                    llm = LLM.get(llm=llm_type, top_p=top_p, top_k=top_k, temperature=temperature, max_tokens=max_new_tokens, OpenAi_Host=OpenAi_Host, OpenAi_Key=OpenAi_Key, OpenAi_Model=OpenAi_Model)
                except ImportError:
                    st.write('Could not import openai python package. Please install it with `pip install openai`')
            else:
                llm_options = [value for dictionary in Weights.get_dict() for value in dictionary.values()]
                local_Model = st.selectbox(label='select llm model', options=llm_options, index=2, disabled=st.session_state.lock_widget)
                if st.button(label='Download model', key='download', disabled=st.session_state.lock_widget, on_click=callback):
                    with st.spinner('please wait, this will take some time...'):
                        Weights.get(model=local_Model)
                    st.session_state.lock_widget = False
                    st.experimental_rerun()
                try:
                    llm = LLM.get(llm=llm_type, top_p=top_p, top_k=top_k, temperature=temperature, max_tokens=max_new_tokens, model_path=f'./Models/{local_Model}')
                except ImportError:
                    st.write('Could not import llama.cpp python package. Please install it with `pip install llama-cpp-python`')
           
        with st.expander(label="condense question prompt template", expanded=False):
            system_name = st.text_input(label='system prompt name', value='### System:', disabled=st.session_state.lock_widget)
            system_text = st.text_area(label='system prompt text', value='You are an AI assistant that follows instruction extremely well. Help as much as you can.', disabled=st.session_state.lock_widget)
            user_name = st.text_input(label='user prompt name', value='### User:', disabled=st.session_state.lock_widget)
            user_text = st.text_area(label='user prompt text', value='Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.', disabled=st.session_state.lock_widget)
            input_name = st.text_input(label='input prompt name', value='### Input:', disabled=st.session_state.lock_widget)
            input_text = st.text_area(label='input prompt text', value='Chat History:\n{chat_history}\nFollow up question: {question}', disabled=st.session_state.lock_widget)
            response_name = st.text_input(label='response prompt name', value='### Response:', disabled=st.session_state.lock_widget)
            response_text = st.text_area(label='response prompt text', value='Standalone question:', disabled=st.session_state.lock_widget)
            Condense_template = Template.get(sys_name=system_name, system=system_text, user_name=user_name, user=user_text, input_name=input_name, input=input_text, res_name=response_name, response=response_text)
            Condense_template_preview = st.code(body=Condense_template)
        
        with st.expander(label="QA prompt template", expanded=False):
            system_name = st.text_input(label='system prompt name', value='### System:', key='qa_sys_name', disabled=st.session_state.lock_widget)
            system_text = st.text_area(label='system prompt text', key='qa_sys', value='You are an AI assistant that follows instruction extremely well. Help as much as you can.', disabled=st.session_state.lock_widget)
            user_name = st.text_input(label='user prompt name', value='### User:', key='qa_user_name', disabled=st.session_state.lock_widget)
            user_text = st.text_area(label='user prompt text', key='qa_user', value='''Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.''', disabled=st.session_state.lock_widget)
            input_name = st.text_input(label='input prompt name', value='### Input:', key='qa_input_name', disabled=st.session_state.lock_widget)
            input_text = st.text_area(label='input prompt text', key='qa_input', value='Question: {question}\nPieces of context:\n{context}', disabled=st.session_state.lock_widget)
            response_name = st.text_input(label='response prompt name', value='### Response:', key='qa_response_name', disabled=st.session_state.lock_widget)
            response_text = st.text_area(label='response prompt text', value='Helpful Answer:', key='qa_response', disabled=st.session_state.lock_widget)
            QA_template = Template.get(sys_name=system_name, system=system_text, user_name=user_name, user=user_text, input_name=input_name, input=input_text, res_name=response_name, response=response_text)
            QA_template_preview = st.code(body=QA_template)

    if prompt := st.chat_input(placeholder="Query", key='user_input', disabled=st.session_state.lock_widget, on_submit=callback):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.spinner('generating response...'):
            if st.session_state.db is None:
                response = 'no retriever present, please process documents first'
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
                    st.markdown(f'##### Source: {source_name}\n')
                    source_data = sources[source]
                    sorted_data = sorted(source_data.items(), key=lambda x: int(x[0]))
                    
                    for page, content in sorted_data:
                        with st.expander(label=f'Page: {page}', expanded=False):
                            st.markdown(content)
        else:
            sources = st.markdown(body='make a query first')
    
    with tab5:
        with st.expander(label='chat_history'):
            st.write(st.session_state['chat_history'])
        with st.expander(label='db'):
            st.write(st.session_state['db'])
                        
if __name__ == '__main__':
    main()
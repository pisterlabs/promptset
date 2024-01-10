# improve_prompt.py

import streamlit as st

from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from functions.best_practices import best_practices_var

MODEL_NAME = ["gpt-4", "gpt-4-1106-preview", "gpt-3.5-turbo-16k", "gpt-3.5-turbo"]
MAX_TOKENS = 2000
DEFAULT_TEMP = 0.25

def clear_text():
    st.session_state["text_improve"] = ""

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", display_method='code'):
        self.container = container
        self.text = initial_text
        self.display_method = display_method

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token 
        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            display_function(self.text)
        else:
            raise ValueError(f"Invalid display_method: {self.display_method}")


def get_improved_prompt(query, chat_box, model_prompt, temp_prompt):
    messages = [
    SystemMessage(
        content=best_practices_var    
    ),
    HumanMessage(
        content=query
        ),
    ]

    stream_handler = StreamHandler(chat_box, display_method='code')
    chat = ChatOpenAI(model=model_prompt, temperature=temp_prompt, max_tokens=MAX_TOKENS, streaming=True, callbacks=[stream_handler])
                      
    try:
        response = chat(messages)
        return response.content
    except Exception as e:
        st.write(f"An error occurred: {str(e)}")
    return None

# Get user input and return improved
def improve_prompt_ui():
    st.markdown(f'<h3 style="border-bottom: 2px solid #3ca0ff; ">{"Improve"}</h3>', 
                unsafe_allow_html=True)
    st.text(" ")

    improve_info = """
    🛠️ Get an improved version of your prompt that follows prompt engineering best practices. Include the relevant column names from your table in double square brackets: [[col_name]].
"""
    html_code = f"""
    <div style="background-color: rgba(244,249,254,255); olor:#283338; font-size: 16px; border-radius: 10px; padding: 15px 15px 1px 15px;">
        {improve_info}
    </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)
    
    st.text(" ")

    with st.chat_message("ai"):
        chat_box = st.markdown("Heyo! I'm here to help you improve the wording of your prompt. Simply type it in the area below and I'll do the rest. 🦾")
    instructions = st.empty()

    with st.container():
        col1, col2 = st.columns([7, 1])
        with col1: 
            with st.chat_message("user"):
                query = st.text_area("User input", placeholder="Example: Create a short fun message informing the customer that the [[product]] will be back in stock on [[date]].", label_visibility="collapsed", key="text_improve")
        with col2: 
            ask_button = st.button("Improve", use_container_width=True)
            reset = st.button("Reset", use_container_width=True, on_click=clear_text)
        with st.expander("__Parameter Settings__"):
            col1, col2, _ = st.columns(3)
            model_prompt = col1.selectbox("Model", MODEL_NAME, help='For best results, the "gpt-4" model is recommended.')
            temp_prompt = col2.slider("Temperature", min_value=0.0, max_value=1.0, value=DEFAULT_TEMP, 
                                        help="Lower values for temperature result in more consistent outputs, while higher values generate more diverse and creative results. Select a temperature value based on the desired trade-off between coherence and creativity for your specific application.", 
            )    
    
    if query and ask_button:
        st.session_state.improved_content = get_improved_prompt(query, chat_box, model_prompt, temp_prompt)
    
    if reset:   
        st.session_state.improved_content = ""

    if st.session_state.improved_content:
        chat_box.code(st.session_state.improved_content, language="http") 
        with instructions.chat_message("ai"):
            st.markdown("""
                        Here is an improved version of your prompt. If you are happy with the wording and example given, simply copy the entire box in the top right corner and use it in the next step. 
                        
                        💡 By providing an example in the prompt, the model will better understand your requirements, returning a more accurate and contextually relevant response.
                        
                        
                        """)
            
def improve_prompt():
    if 'improved_content' not in st.session_state:
        st.session_state.improved_content = ""
    
    improve_prompt_ui()
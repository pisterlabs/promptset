import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

st.title("Translation using Language Models")

styles = [
    "American English in a calm and respectful tone",
    "A polite tone that speaks in English Pirate",
    "A polite tone that speaks in French",
    ]
style_sel = st.selectbox("Selecciona un estilo", options=styles)

text_to_translate = st.chat_input("Text to translate")

with st.expander("Text examples"):
    st.write("""
    Arrr, I be fuming that me blender lid \
    flew off and splattered me kitchen walls \
    with smoothie! And to make matters worse, \
    the warranty don't cover the cost of \
    cleaning up me kitchen. I need yer help \
    right now, matey!

    Hey there customer, \
    the warranty does not cover \
    cleaning expenses for your kitchen \
    because it's your fault that \
    you misused your blender \
    by forgetting to put the lid on before \
    starting the blender. \
    Tough luck! See ya!
    """)

# Define the things that dont change
chat = ChatOpenAI(openai_api_key=st.secrets["OPENAI_KEY"], temperature=0.0)
template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```
{text}
```
"""
prompt_template = ChatPromptTemplate.from_template(template_string)

if text_to_translate:
    translation_request = prompt_template.format_messages(
                    style=style_sel,
                    text=text_to_translate)
    translated_text = chat(translation_request)
    with st.chat_message("user"):
        st.write(text_to_translate)
    with st.chat_message(name="computer", avatar="🤖"):
        st.write(translated_text.content)

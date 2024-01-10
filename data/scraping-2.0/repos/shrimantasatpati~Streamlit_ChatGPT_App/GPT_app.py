import streamlit as st
import openai
from datetime import datetime
from streamlit.components.v1 import html
import webbrowser


st.set_page_config(page_title="ChatGPT App Demo")

html_temp = """
                <div style="background-color:{};padding:1px">
                </div>
                """

url = "https://shrimantasatpati.hashnode.dev/"

with st.sidebar:
    st.markdown("""
    # About 
    ChatGPT App Demo is a primitive tool built on GPT-3.5 to generate ideas on a given topic. This uses the model_engine text-davinci-003. 
    """)
    st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"),unsafe_allow_html=True)
    st.markdown("""
    # How does it work
    Simply enter the topic of interest in the text field below and ideas will be generated.
    You can also download the output as txt.
    """)
    st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"),unsafe_allow_html=True)
    st.markdown("""
    Made by [Shrimanta Satpati](https://shrimantasatpati.hashnode.dev/)
    """,
    unsafe_allow_html=True,
    )
    if st.button('SatpatAI'):
        webbrowser.open_new_tab(url)


input_text = None
if 'output' not in st.session_state:
    st.session_state['output'] = 0

if st.session_state['output'] <=2:
    st.markdown("""
    # ChatGPT Demo
    """)
    input_text = st.text_input("What are you looking for today?", disabled=False)
    st.session_state['output'] = st.session_state['output'] + 1

hide="""
<style>
footer{
	visibility: hidden;
    position: relative;
}
.viewerBadge_container__1QSob{
    visibility: hidden;
}
<style>
"""
st.markdown(hide, unsafe_allow_html=True)

st.markdown(
    """
    <style>
        iframe[width="220"] {
            position: fixed;
            bottom: 60px;
            right: 40px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
if input_text:
    prompt = "What are you looking for today? "+str(input_text)
    if prompt:
        #openai.api_key = st.secrets["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"]
        openai.api_key = "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"
        #response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=150)
        #brainstorming_output = response['choices'][0]['text']
        response = openai.Completion.create(engine="text-davinci-003",
                                            # prompt="Correct this to standard English:\n\nShe no went to the market.",
                                            prompt=prompt,
                                            temperature=0,
                                            top_p=1,
                                            max_tokens=60,
                                            frequency_penalty=0,
                                            presence_penalty=0)
        brainstorming_output = response.choices[0].text
        today = datetime.today().strftime('%Y-%m-%d')
        topic = "What are you looking for today? "+input_text+"\n@Date: "+str(today)+"\n"+brainstorming_output
        
        st.info(brainstorming_output)
        filename = "ChatGPT_response"+str(today)+".txt"
        btn = st.download_button(
            label="Download txt",
            data=topic,
            file_name=filename
        )




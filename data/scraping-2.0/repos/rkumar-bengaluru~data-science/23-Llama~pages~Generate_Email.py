import streamlit as st
import langchain

from utils.utils import get_transformer, get_prompt_template_for_email

def getLLMResponse(form_input, email_sender, email_reciepient, email_style):
    llm = get_transformer()
    template = """
    Write a email with {style} style and includes topic : {email_topic}.\n\nSender: {sender}\n Recipient:{recipient}
    \n\n Email Text:
    """

    prompt = get_prompt_template_for_email(template)
    response = llm(prompt.format(email_topic=form_input, sender=email_sender, recipient=email_reciepient, style=email_style))
    print(response)
    return response


st.header("Generate EMails")
st.write('https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML')
st.write('Example Topic - Tell me about president of singapore')
form_input = st.text_area("Enter Email Topic", height=275)

col1, col2, col3 = st.columns([10, 10, 5])
with col1:
    email_sender = st.text_input("Sender Name")
with col2:
    email_reciepient = st.text_input('Recipient Name')
with col3:
    email_style = st.selectbox("Writing Style",
                               {"Formal", "Appreciating", "Not Satisfied", "Neutral"}, index=0)

submit = st.button("Generate")

if submit:
    st.write(getLLMResponse(form_input, email_sender, email_reciepient, email_style))
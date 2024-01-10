#https://beta.openai.com/docs/guides/embeddings/what-are-embeddings

import openai
import streamlit as st

# st.secrets["pass"] is accessing the value of the "pass" secret.
openai.api_key = st.secrets["api"]

st.header("Demand Letter Creator")
# Create Text Area Widget to enable user to enter texts
text = st.text_area("Enter instructions here")
#st.file_uploader("Upload files")

if len(text) > 100: 
    temp = st.slider("temperature", 0.0, 1.0, 0.5)
    if st.button("Generate"): 
        response = openai.Completion.create(
                    engine = "text-davinci-003",
                    prompt= 'Write a demand letter to our client based on the following set of instructions, communication details to and from, including all dates the client was reached via communication means, events in between these dates, any sums of money transacted/to be transacted (on different between the dates), description of tangible/intangible assets in questions, while maintaining legal jargon.' + text, 
                    #Start the demand letter by stating - We act for [ ] Limited (hereinafter "our Client") and have instructions to address you as hereunder:' + text,
                    max_tokens = 2048,
                    temperature = temp)
        res = response["choices"][0]["text"] 
        st.info(res)
        st.download_button("Download", res)
else: 
    st.warning("Instructions issued are not sufficient")
    

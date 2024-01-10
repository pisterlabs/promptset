import openai
import streamlit as st
  
  #set the GPT api key
openai.api_key=st.secrets['pass']

st.header("Summarizer App using OpenAI+StreamLit")

article_text=st.text_area("Enter text you want to summarize")
output_size = st.radio(label = "What kind of output do you want?", 
                    options= ["To-The-Point", "Concise", "Detailed"])

if output_size == "To-The-Point":
    out_token = 50
elif output_size == "Concise":
    out_token = 128
else:
    out_token = 516

if len(article_text) > 100:
    temp=st.slider("temparature", 0.0, 1.0, 0.5)
    if st.button("Generate Summary"):
       
       #Use GPT-3.5 generating the summary of article
       response=openai.Completion.create(
            engine="text-davinci-003",
           prompt="Please summarize this scientifc article for me in a few sentences : " + article_text,
           max_tokens= out_token,
           temperature=temp
       ) 
       #print the summary of article
       res=response["choices"][0]["text"]
       st.info(res)

       st.download_button ("Download Result", res)
else:
    st.warning('The Sentence is not long enough!')
import streamlit as st
import os
import cohere
from dotenv import load_dotenv
load_dotenv()
api_key = st.secrets["key"]

co = cohere.Client(api_key)
st.title("Blog Content Generator")
st.markdown(
    "This app generates blog-post style text using [Cohere's Large Language Model](https://docs.cohere.com/). You can find the code for this app on [GitHub](https://github.com/cechae/blog-content-generator)."
)


with st.form("my_form_3"):
    tone = st.selectbox("Tone(Optional)", ("Friendly","Casual", "Formal", "Persuasive", "Informative", "Funny", "Serious", "Clever", "Creative", "Boring"))
    creativity = st.slider(label="Creativity",min_value=0.0,max_value=3.0, value=1.0, step=.25)
    subject = st.text_area("What should I write about?")
    submitted = st.form_submit_button("Submit")

n_gens=1
freq=0.5
def generate(prompt,model_size="xlarge",n_generations=n_gens, temp=.75, tokens=250, stops=["--"], freq=freq):
    prediction = co.generate(
                    model=model_size,
                    prompt=prompt,
                    return_likelihoods = 'GENERATION',
                    stop_sequences=stops,
                    max_tokens=tokens,
                    temperature=temp,
                    num_generations=n_generations,
                    k=0,
                    frequency_penalty=freq,
                    p=0.75)
    return(prediction)

def max_likely(pred):
    
    return pred.generations[0].text
    
with st.spinner('Generating Content...'):
    if submitted:
        
        
        content_prompt = "Write a blog about \n\n " + subject + "\n\n" +  "Write it in a " + tone + " tone."
        content_prediction = generate(prompt=content_prompt, model_size='command-nightly', temp=creativity, tokens=300, stops=["----"])
        content = max_likely(content_prediction)
        
        title_prompt = "Write a creative title for this blog. \n\n" + "Blog:" + content + "\n\nTitle:"
        title_prediction = generate(prompt=title_prompt, model_size='command-nightly', temp=creativity, tokens=25, stops=["---"])
        title = max_likely(title_prediction)
        
        st.header(title)
        st.write(content)
    
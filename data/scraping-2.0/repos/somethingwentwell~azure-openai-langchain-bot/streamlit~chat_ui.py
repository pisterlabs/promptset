import os  
import openai  
import streamlit as st  
  
# Set up OpenAI API  
openai.api_type = "azure"  
openai.api_base = "https://fevaworksopenai.openai.azure.com/"  
openai.api_version = "2023-03-15-preview"  
openai.api_key = os.getenv("OPENAI_API_KEY")  
  
# Streamlit app  
st.title("AI Chatbot")  
st.write("Ask your question and get a response from the AI.")  
  
user_input = st.text_input("Your question:")  
if st.button("Send"):  
    if user_input:  
        response = openai.ChatCompletion.create(  
            engine="gpt-35-turbo",  
            messages=[{"role": "system", "content": "You are an AI assistant that helps people find information."},  
                      {"role": "user", "content": user_input}],  
            max_tokens=800,  
            top_p=0.95,  
            frequency_penalty=0,  
            presence_penalty=0,  
            stop=None)  
  
        st.write("AI response:", response.choices[0].message["content"])  
    else:  
        st.write("Please enter a question.")  

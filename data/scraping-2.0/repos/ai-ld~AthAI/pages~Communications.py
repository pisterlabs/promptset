import openai
import streamlit as st
#import pinecone
import os

from langchain.document_loaders import WebBaseLoader, PyPDFLoader
#from langchain.embeddings.openai import OpenAIEmbeddings
#from langchain.vectorstores import Pinecone
#from langchain.document_loaders import UnstructuredFileLoader

#index_name = "new-demo"
#docsearch = None

# initialize pinecone
#pinecone.init(
#    api_key=st.secrets["pinecone-key"],  # find at app.pinecone.io
#    environment=st.secrets["pinecone-env"]  # next to api key in console
#)

# Set up page configuration
st.set_page_config(page_title="Communications 📣", page_icon=":mega:", layout="wide")

# Set up page header and subheader
st.title("Political Campaign AI-powered Communication Tool")
st.subheader("A custom app for generating communications")

# Set API key
openai.api_key = st.secrets["oai-key"]
os.environ["OPENAI_API_KEY"] = st.secrets["oai-key"]

#embeddings = OpenAIEmbeddings()

# Description and information
st.write("This tool generates communications for political campaigns using OpenAI's GPT-3 service. "
         "Please enter as much information as you can, and GPT will handle the rest.\n\n"
         "Note: GPT-3 might generate incorrect information, so editing output is still necessary. "
         "This is a demo with limitations.")

# Create a tab selection
tabs = st.selectbox(
    'Which communication do you want to create? 📄',
    ('Email 📧', 'Press Release 📰', 'Social Media 📲'))

# Function to generate content using GPT
def generic_completion(prompt):
    completions = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.85
    )
    message = completions['choices'][0]['message']['content']
    return message.strip()

# Function to generate a tweet
def tweet(output):
    return generic_completion(
        "Generate a tweet summarizing the following text. "
        "Make it engaging and concise: " + output)

# Email tab
if tabs == 'Email 📧':
    subject = st.text_input("Email subject:")
    recipient = st.text_input("Recipient:")
    details = st.text_area("Email details:")

    if st.button(label="Generate Email"):
        try:
            output = generic_completion("Generate a well-written and engaging email for a political campaign. "
                                        "The email is to be sent to " + recipient + " with the subject " + subject +
                                        ". The email should include the following details: " + details)
            st.write("```")
            st.write(output)
            st.write("```")
        except:
            st.write("An error occurred while processing your request.")

# Press Release tab
elif tabs == 'Press Release 📰':
    body = st.text_area("Press release content:")
    if st.button(label="Generate Press Release"):
        try:
            output = generic_completion(f"Generate a compelling press release for a political campaign. "
                                        "The press release should be based on the following: " + body)
            st.write("```")
            st.write(output)
            st.write("```")
        except:
            st.write("An error occurred while processing your request.")

# Social Media tab
elif tabs == 'Social Media 📲':
    platform = st.selectbox('Select Social Media Platform', ['Twitter 🐦', 'Facebook 👍', 'Instagram 📷', 'LinkedIn 🔗'])
    category = st.selectbox("Choose a topic:", ["Campaign Announcement 📢", "Policy Position 📚", "Event Invitation 🎟️", "Fundraising 💰"])
    info = st.text_input("Details", "")
    if st.button(label="Generate Social Media Post"):
        prompt = f"Generate an engaging {platform} post for a political campaign on the topic: {category} with details: {info}"
        #prompt = generate_prompt_with_similar_docs(base_prompt, category, loaded_data)
        generated_post = generic_completion(prompt)
        st.write(generated_post)

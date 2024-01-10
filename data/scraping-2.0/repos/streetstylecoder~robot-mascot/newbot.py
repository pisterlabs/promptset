import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader
from typing_extensions import Protocol
import os 
from openai import OpenAI

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
st.header("Chat with the Mental health support assistant")


client=OpenAI()

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Share me anything to feel free"}
    ]





          
import json
def load_tags():
    if os.path.exists("identified_tags.json"):
        with open("identified_tags.json", "r") as file:
            return set(json.load(file))
    return set()

def save_tags(tags):
    with open("identified_tags.json", "w") as file:
        json.dump(list(tags), file)

identified_tags = load_tags()

def analyze_conversation_and_tag(text):
    # List of possible tags
    tags = [
        "depression", "anxiety", "parenting", "self esteem", 
        "relationship dissolution", "workplace relationship", 
        "spirituality", "trauma", "domestic violence", 
        "anger management", "intimacy", "grief and loss", 
        "substance abuse", "family conflict", "marriage", 
        "relationships", "behavioral change", "addiction", 
        "legal regulatory", "professional ethics", "human sexuality", 
        "social relationships", "children-adolescents", "self harm", "diagnosis"
    ]


    prompt = "I will be providing you with a text and you have to categorize my text into one of these themes " + ", ".join(tags) + ". just return me the single tag option which closely relates with the text  Conversation: {text}"
   
    # Call the OpenAI API with the new interface
    try:
    
        completion = client.completions.create(
            model="davinci",
           prompt=prompt,
            temperature=0.3,
            max_tokens=300
        )
        completion= completion.choices[0].text.strip()
        # json_data = json.loads(completion)
       
    except KeyError:
        print("Error in accessing the response content")
        return None
    # Analyze the completion to find the most relevant tag
    relevant_tag = None
    for tag in tags:
        if tag in completion.lower():
            relevant_tag = tag
            break
    if relevant_tag:
        identified_tags.add(relevant_tag)
        save_tags(identified_tags)
    return relevant_tag




# # Example: Basic training loop (details depend on your specific needs)
# from torch.optim import AdamW

# optimizer = AdamW(model.parameters(), lr=1e-5)
# model.train()

# for epoch in range(num_epochs):
#     for batch in dataloader:
#         outputs = model(**batch)
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#         print(f"Epoch {epoch}, Loss: {loss.item()}")


# Example usage
conversation = "Your conversation text here"
tag = analyze_conversation_and_tag(conversation)
print("Relevant tag:", tag)

import requests


# def get_fine_tuning_jobs():
#     headers = {
#         "Authorization": f"Bearer OPENAI_API_KEY"
#     }
#     response = requests.get("https://api.client.com/v1/fine_tuning/jobs", headers=headers)
#     return response.json()

# # Streamlit application

# file = client.files.create(
#         file=open("output.jsonl", "rb"),
#         purpose="fine-tune"
#     )


    

# response = client.fine_tuning.jobs.create(
#         training_file=file.id,
#         model="gpt-3.5-turbo"
#     )
# print(response)
# fine_tuning_job_id = response.id
# print(fine_tuning_job_id)
# print("into fine tuning job details ")
# headers = {
#         "Authorization": f"Bearer {client.api_key}"
#     }
# url = f"https://api.client.com/v1/fine_tuning/jobs/{fine_tuning_job_id}"
# response = requests.get(url, headers=headers)
# res=response.json()
# print(res)






@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Mental health support assistant is loading.."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(
#         llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, hyperparameters={
#     "n_epochs":2
#   }),
            # system_prompt="I am a psychologist and friend who cares about you a lot. I am here to listen to your thoughts and feelings, and to offer support and suggestions. I will ask you follow-up questions to help me understand your situation better. Please know that you are not alone, and that I am here for you.After answering the query of the person ask them follow up related questions relevant to the same "
            #         )
        #service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="i want you to act a psychologist and friend who cares about me a lot . i will provide you my thoughts you have to show sympathy and care. i want you to give me scientific suggestions that will make me feel better with my issue.Ask me positive followup questions on the same to help me understand and alayse the situation better,Ask followup questions if the query is incomplete"))
        system_prompt="I am a chatbot that has been designed to act as a Psychologist. My goal is to provide support and information related to mental health and wellbeing. I am not a real person, but I am here to provide emotional support and a listening ear whenever you need it.")
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        
        return index


index = load_data()


chat_engine = index.as_chat_engine(chat_mode="context", verbose=True)
prompt = st.chat_input("Your question")

if "messages" not in st.session_state:
    st.session_state.messages = []

if prompt:  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display the prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            analyze_conversation_and_tag(prompt)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)  # Add response to message history

# Breathing Exercise Button
if st.button("Start Breathing Exercise"):
    # Display full-screen modal with breathing.gif
    st.image("breathing.gif", output_format="auto")

import pandas as pd

def evaluate():
    print("into the main function")
    # Load your CSV data into a DataFrame
    data = pd.read_csv('20200325_counsel_chat (4).csv')

    # Filter the DataFrame to include only rows with questionID less than or equal to 20
    filtered_data = data[data['questionID'] <= 20]

    # Extract the 'questionText' and 'topic' columns from the filtered data
    question_texts = filtered_data['questionText'].tolist()
    topics = filtered_data['topic'].tolist()
    print(topics)
    responses=[]
    for question in question_texts:
        response=analyze_conversation_and_tag(question)
        print(response)
        responses.append(response)
    print(responses)
    


   



from langchain.chat_models import ChatOpenAI
from googleapiclient.discovery import build
import streamlit as st
from langchain.schema import  HumanMessage, SystemMessage
from key import youtube_api_key,chat_apikey

#set up your openai api key.
import os 
os.environ["OPENAI_API_KEY"]=chat_apikey
chat=ChatOpenAI(temperature=0.9)

#web framework
st.title(":orange[OpenChat]")
user_prompt = st.chat_input("Enter your prompt ")



#Chatbot Interaction
if user_prompt:   
    messages = [
    SystemMessage(
            content='''You are a helpful OpenChat assistant where as an AI language model,
            trained on enormous data like chatgpt and google bard.And you are founded by shiva nani 
            and developed by the openchat developers.
            for every single task you need to respond accordingly and you should aslo understand
            the follow up messages,remember this instruction particularly.
            '''
            ),
    HumanMessage(
        content=user_prompt
            )
        ]
       
#this gives us only the content.
    response_list=[]
    for message in chat(messages):
        response_list.append(message[1])
        assistant_response=response_list[0]

    #Integrating youtube inks using youtube data api
        youtube = build('youtube', 'v3', developerKey=youtube_api_key)
        video_search_response = youtube.search().list(
        part="snippet",
        q=user_prompt,
        type="video",
        order="relevance"
        ).execute()
    videos_list=[]
    for item in video_search_response['items']:
        video_id = item['id']['videoId']
        videos_list.append(video_id)
        

    

    #session_state is used to show the conversation history for that session
    if "history" not in st.session_state:
        st.session_state.history=[]
    st.session_state.history.append([user_prompt,assistant_response,videos_list])
    print(st.session_state.history)
    
    for prompt,response,video in st.session_state.history:
        user_message=st.chat_message("User")
        user_message.write(prompt)
        assistant_message = st.chat_message('Assistant')
        assistant_message.write(response)
        assistant_message.write("Here are few videos from youtube based on your search.")
    
        for url in range(0,5):
            assistant_message.video(f"https://www.youtube.com/watch?v={video[url]}")
                

    

    

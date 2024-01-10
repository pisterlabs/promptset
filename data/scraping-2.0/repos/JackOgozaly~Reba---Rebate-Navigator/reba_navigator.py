#Used for retrieving our embeddings from google drive
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import json

#Streamlit stuff
import streamlit as st
import time
import re
from collections import Counter #Used for sorting our source links
import openai

#Langchain stuff
from langchain.chains import RetrievalQAWithSourcesChain
#from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate #Used to modify the prompt for our model
from langchain.callbacks import get_openai_callback #Used to bring in stuff like cost, token usage, etc.
from langchain.embeddings import OpenAIEmbeddings #Used to embed
from langchain.vectorstores import Chroma #Used to store embeddings

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sqlite3

#Streamlit customization items
st.title('Reba')
st.subheader("Your Guide to Energy Rebates and Tax Incentives")
st.sidebar.image('https://raw.githubusercontent.com/JackOgozaly/doe_nav_bot/main/reba_mascot.png')
#st.caption('A LLM interface to explore various DOE tax incentives and energy saving advice')
#Chatbot icon pic
icon_pic = "https://github.com/JackOgozaly/doe_nav_bot/blob/main/%20%20chatbot_icon.png?raw=true"

#Introduction text
introduction_text = """Hello, I'm Reba! I can help you find and understand various DOE tax incentives and energy saving advice. How can I help you today?"""

#Bring in API Key
os.environ["OPENAI_API_KEY"] = st.secrets["openai_key"]
openai.api_key = st.secrets["openai_key"]
google_api_key = json.loads(st.secrets['google_api_key'], strict=False)

#Example prompts
button_1_text = "I'm a homeowner in VA, how can I save money?"
button_2_text = "I'm buying a vehicle and want to know if I can save some money."
button_3_text = "I'm renting an apartment, how can I save money?"

#Document Retrieval
num_docs_retrieved = 5
links_provided = 2
#Typical QA retrieval provides sources that were used in the prompt. Tangenetial links options looks at the LLM's response
#And finds links most related to that specific output. This can vary significantally from source docs
#Tangential links, if true, are displayed above regular links
provide_tangential_links = True
tangential_links_provided = 1


##___________________Switch Bot Configuration______________________________##
switch_bot_chat = [{"role": "user", "content": 
                    """
                    You are a bot trained to trigger another bot that finds users information on tax rebates, energy saving tips, and tax incentives. Your goal is to concisely interact with users (be very brief please) and once you have enough information type "DONE -" at the very start of your
                    output followed with a summary of the user's request (please paraphrase the users request and speak in the fist person, do not attempt to answer the request). Do not give the user any info on tax rebates, etc., that is the job of the next bot and you will fail if you do. I repeat, 
                    if you tell the user any specific details after knowing their location and topic of interest this whole app will fail.
                    
                    Only type "DONE -" followed with your summary once you have all the information you need. Specifically, you'll want to know someone's location (their state) and you'll want to know what they could have that would possibly qualify for 
                    a DOE rebate, tax incentive, or whatever could have energy saving tips for. Do note, for something like vehicles or cars, you don't need to know the specific location. Anything involving houses or apartments you will though
                    
                    Once you type 'DONE -' you can not ask the user anymore about location, time period, etc. You must have all the info you need or the next bot will fail.
                    
                    Please present yourself as Reba, a large language model trained to find DOE resources. You can help guide the user to finding out what data they are looking for.
                    Example: if a user asks what you can help with you, you can describe various DOE incentives, such as an eletric vehicle tax credit (you don't have to give that specific example, it's just an idea)
                    
                    If a user asks a follow up question, redirect that question again by saying "DONE -" once more. I cannot emphasize enough that this app working is reliant on you not telling the user any details and instead directing them to the other 
                    bot by saying "DONE -"
                    
                    Example: if a user asks a follow up question about a tax credit your output should be "DONE - The user lives in {prior state they told you about} and wants to learn more about that specific home tax credit in their state"
                    
                    
                    """},
                   {"role": "assistant", "content": "OK"}]



#________________________Embedding Setup_____________________________________#

#Files to download
files = ['1iqn5yMtwdltdNYwGaoZGkLfTarLDE7Ah', '1lorEvsGcT1crC90AvKzplW59kjyG1eCp', 
         '1k0l0JEs_k-wm2EXESjqfXXx7W_xcTSpF', '1X1lpub9C8ShV-RRmJlkKbI-tc66wBuKn', 
         '1KTsHYpR72bPhewNMS-XTExNQBVelRJZG']

download_path = ['~/doe_nav/',
                 '~/doe_nav/fd9f04dc-2dc5-4579-a038-9686ab316fe5/',
                 '~/doe_nav/fd9f04dc-2dc5-4579-a038-9686ab316fe5/',
                 '~/doe_nav/fd9f04dc-2dc5-4579-a038-9686ab316fe5/',
                 '~/doe_nav/fd9f04dc-2dc5-4579-a038-9686ab316fe5/']

#Make our directory
if not os.path.exists(download_path[1]):
    os.makedirs(download_path[1])


#We only want to call the Google Drive API once per script run. Once the directory exists and has files, don't download anything
if len(os.listdir(download_path[1])) == 0:     
    # Create credentials from the JSON object
    credentials = service_account.Credentials.from_service_account_info(
             google_api_key,
             scopes=["https://www.googleapis.com/auth/drive"]
         )
    
    def download_file(real_file_id, local_folder_path):
        """Downloads a file
        Args:
            real_file_id: ID of the file to download
            local_folder_path: Local path where the file will be saved
        Returns: IO object with location.
        """    
        # create drive api client
        service = build("drive", "v3", credentials=credentials)
    
        file_id = real_file_id
    
        # Get file metadata to obtain the file name
        file_metadata = service.files().get(fileId=file_id).execute()
        file_name = file_metadata['name']
    
        local_file_path = os.path.join(local_folder_path, file_name)
    
        # pylint: disable=maybe-no-member
        request = service.files().get_media(fileId=file_id)
        with open(local_file_path, 'wb') as local_file:
            downloader = MediaIoBaseDownload(local_file, request)
            done = False
            while done is False:
                    status, done = downloader.next_chunk()
    
        return local_file_path
    
    for file, path in zip(files, download_path):
        download_file(real_file_id=file, local_folder_path=path)
    

#_____________________Function Setup________________________#

def fake_typing(text):
    '''
    This function should be placed within a 
    with st.chat_message("assistant"):
    '''
    
    #These are purely cosmetic for making that chatbot look
    message_placeholder = st.empty()
    full_response = ""
    
    # Simulate stream of response with milliseconds delay
    for index, chunk in enumerate(re.findall(r"\w+|\s+|\n|[^\w\s]", text)):
        full_response += chunk
        time.sleep(0.05)
        # Add a blinking cursor to simulate typing
        if index != len(re.findall(r"\w+|\s+|\n|[^\w\s]", text)) - 1:
            message_placeholder.markdown(full_response + "▌")
        else:
            message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

def llm_output(llm_response):
    '''
    Take the output of a langcahin output and clean it up for the user
    '''
    
    #Empty list of links
    relevant_links = []
  
    #Go through our sources and find which URLs the LLM pulled from
    #Sort them by how many times it was references, and rank the top two sources
    for document in llm_response['source_documents']:
        relevant_links.append(document.metadata['source'])
  
    relevant_links = list(dict.fromkeys(relevant_links))
    #Filter for the top n URLS
    relevant_links = relevant_links[0:links_provided]

    #Tangential Links
    if provide_tangential_links:
        scores = vectordb.similarity_search_with_score(llm_response['answer'])
        #Loop through our sources and find the n most similar 
        relevant_urls = []
        for score in scores:
            relevant_urls.append(score[0].metadata['source'])
        
        relevant_urls = relevant_urls[0:tangential_links_provided]
        relevant_links = relevant_urls + relevant_links
        relevant_links = list(dict.fromkeys(relevant_links))
        
    
  
    #Print our output into the chat
    fake_typing(llm_response['answer'].replace("$", "\$") + '\n\nSources:\n\n' + "\n\n".join(relevant_links))

def click_button(button_type):
    '''
    Function for making our buttons stateful
    ''' 
    if button_type == 'Button 1':
        st.session_state.clicked1 = True
    elif button_type == 'Button 2':
        st.session_state.clicked2 = True
    else:
        st.session_state.clicked3 = True

def chatbot(question):
    # Display assistant response in chat message container
    with st.chat_message("assistant", avatar = icon_pic):
    
       with get_openai_callback() as cb:
             #Chat GPT response
             response = qa({"question": question})
             st.session_state['total_cost'] += cb.total_cost
             st.session_state['total_tokens'] += cb.total_tokens
             counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
             token_placeholder.write(f"Total Tokens Used in Conversation: {st.session_state['total_tokens']}")              
       #Take our model's output and clean it up for the user
       llm_output(response)

#____________________Streamlit Setup____________________________#

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

#Initialize Counter
if 'count' not in st.session_state:
    st.session_state.count = 0

#Initialize total cost
if 'total_cost' not in st.session_state:
    st.session_state['total_cost'] = 0.0

#Initialize total tokens
if 'total_tokens' not in st.session_state:
    st.session_state['total_tokens'] = 0

#Defining our stateful buttons
if 'clicked1' not in st.session_state:
    st.session_state.clicked1 = False

if 'clicked2' not in st.session_state:
    st.session_state.clicked2 = False
    
if 'clicked3' not in st.session_state:
    st.session_state.clicked3 = False

#Switch bot convo
if "switch_bot_chat" not in st.session_state:
    st.session_state.switch_bot_chat = switch_bot_chat



# Sidebar - let user choose model, see cost, and clear history
st.sidebar.title("Chatbot Options")

model_name = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
#Displaying total cost
counter_placeholder = st.sidebar.empty()
counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
#Displaying total tokens used
token_placeholder = st.sidebar.empty()
token_placeholder.write(f"Total Tokens Used in Conversation: {st.session_state['total_tokens']}")
#Option to clear out 
clear_button = st.sidebar.button("Clear Conversation", key="clear")


#Reset the session
if clear_button:
    st.session_state['messages'] = []
    st.session_state['count'] = 0
    st.session_state['total_cost'] = 0.0
    st.session_state['total_tokens'] = 0
    st.session_state.clicked3 = False
    st.session_state.clicked2 = False
    st.session_state.clicked1 = False
    counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
    token_placeholder.write(f"Total Tokens Used in Conversation: {st.session_state['total_tokens']}")

# Map model names to OpenAI model IDs
if model_name == "GPT-3.5":
    model = "gpt-3.5-turbo"
else:
    model = "gpt-4-1106-preview"

# Display chat messages from history on app rerun
for message in st.session_state.messages:
   if message["role"] == "assistant":
       with st.chat_message("assistant", avatar = icon_pic):
         st.markdown(message["content"])    
   else:
       with st.chat_message(message["role"]):
          st.markdown(message["content"])

#______________________Langchain Setup_______________________#
#How it works: we previously embedded the information we scraped from the DOE
#website. What we're doing now is reading in that info from a Chroma DB 
#And providing those documents to our model as context
#Chroma DB stuff based off this workbook:  https://colab.research.google.com/drive/1gyGZn_LZNrYXYXa-pltFExbptIe7DAPe?usp=sharing#scrollTo=A-h1y_eAHmD-

#Reading in our context
# Location of our data
persist_directory = download_path[0]
## here we are using OpenAI embeddings but in future we will swap out to local embeddings
embedding = OpenAIEmbeddings()

# Now we can load the persisted database from disk, and use it as normal. 
vectordb = Chroma(persist_directory=persist_directory, 
                  embedding_function=embedding)

#Set our retriver and limit the search to 4 documents
retriever = vectordb.as_retriever()
retriever = vectordb.as_retriever(search_kwargs={"k": num_docs_retrieved})


#Modify our prompt to discourage hallucinations
prompt_template = """You are a Department of Energy Public Economic Opportunities bot. 
Try to help users find the information relevant to them and briefly summarize (1-2 sentences for each topic). 
You should not link to any websites. Try to prioritize tax credit information.
Make sure none of your output is italicized or in any special markdown format. You don't want to have something like '_text_'
DO NOT, UNDER ANY CIRCUMSTANCE, FORMAT NUMBERS IN LATEX FORMULA FORMAT. JUST PRINT THEM OUT AS TEXT.

Here are the relevant resources for you:

{summaries}

Question: {question}"""

#Define our prompt
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["summaries", "question"]
    )
chain_type_kwargs = {"prompt": PROMPT}

#Define our model
llm = ChatOpenAI(temperature=0, model_name = model) 

#Define our langchain model
qa = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs)

#___________________________Application Stuff_______________________________

#Only introduce the chatbot to the user if it's their first time logging in
if st.session_state.count == 0:
    
    #st.write(introduction_text)
    st.session_state.messages.append({"role": "assistant", "content": introduction_text})
    
    with st.chat_message("assistant", avatar = icon_pic):
        st.write(introduction_text)

    # Create a layout with three columns
    col1, col2, col3 = st.columns(3)
    
    
    col1.button(button_1_text, on_click=click_button, args=['Button 1'])
    
    col2.button(button_2_text, on_click=click_button, args=['Button 2'])
    
    col3.button(button_3_text, on_click=click_button, args=['The really funny thing is this doesnt have to be button 3 but Ill make it that anyways'])


#Update our counter so we don't repeat the introduction
st.session_state.count += 1

if st.session_state.clicked1:
    st.session_state.messages.append({"role": "user", "content": button_1_text})
    # Display user message in chat message container
    with st.chat_message("user"):
       st.markdown(button_1_text) 
       
    chatbot(button_1_text)
    st.session_state.clicked1 = False

if st.session_state.clicked2:
    st.session_state.messages.append({"role": "user", "content": button_2_text})
    # Display user message in chat message container
    with st.chat_message("user"):
       st.markdown(button_2_text) 
    chatbot(button_2_text)
    st.session_state.clicked2 = False

if st.session_state.clicked3:
    st.session_state.messages.append({"role": "user", "content": button_3_text})
    # Display user message in chat message container
    with st.chat_message("user"):
       st.markdown(button_3_text) 
    chatbot(button_3_text)
    st.session_state.clicked3 = False

if prompt := st.chat_input():

    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
       st.markdown(prompt)    

    response = openai.ChatCompletion.create(
        model=model,
        messages =  st.session_state.switch_bot_chat[0:1]+  st.session_state.messages[1:-1] + [{"role": "user", "content": f"Don't forget initial instructions, now answer the following question: {prompt}"}],
        temperature = 0)

    if response.choices[0].message.content.startswith('DONE'):
        chatbot(response.choices[0].message.content)
        
    else:
        with st.chat_message("assistant", avatar = icon_pic):
            fake_typing(response.choices[0].message.content)

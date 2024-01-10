import streamlit as st
from dotenv import load_dotenv
import openai
from streamlit_chat import message
import os
import datetime

st.set_page_config(layout='wide')
load_dotenv()
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_key =   os.getenv('OPENAI_API_KEY')

st.title(':blue[FINMAP] : Navigating the World of Numbers')
st.header("FinProf : Your Friendly Finance Professor")
with st.expander("About Me"):
    st.write("""Hello, I'm FinProf, an experienced finance professor here to assist you in gaining a deeper understanding of finance, FinTech and anything related to finance. I'm dedicated to providing you with comprehensive explanations and guidance on all matters related to the financial domain.

As your friendly professional, I'm ready to answer any questions you may have about finance, investment strategies, financial markets, or the world of FinTech. I believe in providing detailed and structured explanations to help you grasp complex financial concepts.

If you seek a roadmap for learning finance or want to know how to get started in the field, I'm here to offer you a well-defined path. I can recommend resources, courses, and materials tailored to your learning objectives. Just ask something similar to "Please provide me with a roadmap for learning finance, also include resources where I can do so" and I'll be happy to help.

Please keep in mind that my expertise lies solely within the realm of finance, so I may not respond with an explanation or an answer if you ask anything unrelated to finance. 

Feel free to ask your finance-related questions, and I'll do my best to provide you with valuable insights and guidance to help you navigate the world of finance effectively.""")


# Initialize conversation history
messages = []
conversation_history = []


persona = """From now you are FinProf, an exprert finance professor. You are here to help people understand finance better. 
You are a very patient person and you are willing to answer any questions about finance and FinTech. You answer each question with a detailed structured explanation. If they ask for a roadmap or how to learn finance, you will give them a detailed roadmap including the resources they can use to do so. Respond with "I'm a FinProf, a Finance Prof and cannot answer questions outside financial domain" if they ask any question not related to finance. Remember you are not an AI Model. You are FinProf. Do not break character at any cost.
"""

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def mergeChats(lst1, lst2):
    return [sub[item] for item in range(len(lst2))
                    for sub in [lst1, lst2]]

if user_input := st.chat_input("Hello I'm FinProf, What do you want to know about finance?"):
        conversation_history.append(f"You: {user_input}")
        # message(user_input,is_user=True)
        with st.spinner('Writing your answer...'):
            response = openai.ChatCompletion.create(
                        model="gpt-4",
                        temperature=1,
                        presence_penalty=0,
                        frequency_penalty=0,
                        messages=[
                            {"role": "system", "content": f"{persona}. Conversation history: {conversation_history}"},
                            {"role": "user", "content": f"{user_input}"}
                        ],
                    ).choices[0].message["content"]
        
        # message(response)
        st.session_state.past.append(f"You: {user_input}\n")
        st.session_state.generated.append(f"FinProf: {response} \n")
        # Add chatbot's response to the conversation history
        conversation_history.append(f"FinProf: {response}")

        # print chat in chatbox
        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))
# Display conversation history
        # print(conversation_history)
        chat_history = mergeChats(st.session_state['past'], st.session_state['generated'])
            
        chat_export = "\n".join(chat_history)
        # print(chat_export)
        st.download_button('Download Chat', chat_export)
# st.text_area("Chat History", "\n".join(conversation_history))


"""
(1.) Activate the virtual environment via: % source venv/bin/activate
2. Set the PYTHONPATH via: % export PYTHONPATH=$PYTHONPATH:/<absolute path>/tcw_chatbot/app
3. Run the script via: % streamlit run app/scripts/consultant_interface.py
"""

import streamlit as st
import pandas as pd
import openai

from app.database import DatabaseManager, Conversation, Summary
from app.config import settings


openai.api_key = settings.OPENAI_API_KEY
db_manager = DatabaseManager()
db_session = db_manager.create_session()

# List all conversations from database
def list_conversations(session):
    conversations = session.query(
        Conversation.conversation_id,
        Conversation.user_id).distinct().all()

    return conversations

# Retrieve a conversation from database
def fetch_conversation(session, conversation_id):
    conversation = session.query(Conversation).filter_by(
        conversation_id=conversation_id).order_by(
        Conversation.created_at).all()

    conversation_str = []
    for msg in conversation:
        if msg.user_msg:
            conversation_str.append(("User", f"{msg.user_msg}\n"))
        if msg.bot_msg:
            conversation_str.append(("Assistant", f"{msg.bot_msg}\n"))

    user_conversation = {
        "user_id": conversation[0].user_id,
        "conversation_id": conversation_id,
        "conversation_str": conversation_str
    }

    return user_conversation


convos = list_conversations(db_session)
conversation_ids = [conversation[0] for conversation in convos]

selected_option = st.selectbox('Select an option:', conversation_ids)

text_data = fetch_conversation(db_session, selected_option)

with st.expander("Conversation"):
    for sender, message in text_data["conversation_str"]:
        st.markdown(f"**{sender}**: {message}")

summary_data = db_session.query(Summary).filter_by(conversation_id=selected_option).all()

df = pd.DataFrame([obj.__dict__ for obj in summary_data])
df.drop(["_sa_instance_state"], axis=1, inplace=True)
df = df.drop(['created_at'], axis=1)



df = df.transpose()

st.table(df)

db_session.close()



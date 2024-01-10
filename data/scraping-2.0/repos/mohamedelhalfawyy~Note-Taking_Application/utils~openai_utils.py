# app/utils/openai_utils.py

import openai
from fastapi import Depends, HTTPException
from database.models import Note
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
import os

def get_openai_api_key():
    return os.getenv("OPENAI_API_KEY")

def summarize_note(db, note_id, openai_api_key):
    db_note = db.query(Note).filter(Note.id == note_id).first()

    if db_note is None:
        raise HTTPException(status_code=404, detail="Note not found")

    openai.api_key = openai_api_key

    template = """You are a helpful assistant specialized in summarizing notes.
    A user will pass in a piece of text, and you should generate a concise summary of that text.
    ONLY return the summary, and nothing more."""
    human_template = "{text}"

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", human_template),
    ])
    chain = chat_prompt | ChatOpenAI()

    response = chain.invoke({"text": db_note.content})
    # Extract the content from the response
    summary = response.content

    # Return only the summary
    return {"summary": summary}

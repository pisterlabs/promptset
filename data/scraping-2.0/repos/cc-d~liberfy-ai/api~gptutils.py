import openai
import os
from sqlalchemy.orm import Session
from typing import Union, Optional, Tuple

from database import add_commit_refresh
from schemas import (
    GPTCreateCompReq,
    BaseChat,
    BaseComp,
    GPTChoice,
    GPTChunkComp,
    GPTComp,
    GPTDelta,
    GPTUsage,
    DBComp,
    DBChat,
    DBMsg,
    DBUser,
)
from models import Chat, Completion, Message, User

openai.api_key = os.getenv('OPENAI_API_KEY')


def submit_gpt_comp(comp: DBComp, db: Session) -> DBComp:
    """Submit a GPTComp to OpenAI and save the result to the database."""

    # Prepare messages for API call
    messages = [{"role": msg.role, "content": msg.content} for msg in comp.messages]

    # Send the request to OpenAI
    response = openai.ChatCompletion.create(
        model=comp.model, messages=messages, temperature=comp.temperature
    )

    # Extract choice from the response
    choice = response['choices'][0]

    # Save the message associated with this choice in the database
    new_message = Message(
        role="assistant", content=choice['message']['content'], completion_id=comp.id
    )
    db.add(new_message)

    # Commit the changes to the database
    db.commit()

    # Refresh the comp object to reflect the new message
    db.refresh(comp)

    # Return the updated comp instance
    return comp

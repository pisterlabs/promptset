from typing import List

from fastapi import APIRouter, HTTPException
from sqlmodel import Session

from pages import engine
from pages.text_completer.v1.models import TextCompletionRead, TextCompletionCreate, TextCompletion

router = APIRouter()

@router.get("/text_completions/")
def get_text_completions() -> List[TextCompletionRead]:
    with Session(engine) as session:
        completions = session.query(TextCompletion).all()
        return [TextCompletionRead(id=c.id, input=c.input, output=c.output) for c in completions]


import openai

# Ensure you set your OpenAI API key

@router.post("/text_completions/", response_model=TextCompletionRead)
def create_text_completion(text_completion: TextCompletionCreate) -> TextCompletionRead:
    with Session(engine) as session:
        # Call OpenAI's chat completion for the input using GPT-4
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text_completion.input}
            ]
        )

        output = response.choices[0].message['content'].strip()

        new_completion = TextCompletion(**text_completion.dict(), output=output)
        session.add(new_completion)
        session.commit()
        session.refresh(new_completion)
        return TextCompletionRead(id=new_completion.id, input=new_completion.input, output=new_completion.output)




@router.get("/text_completions/{completion_id}/", response_model=TextCompletionRead)
def get_text_completion(completion_id: int) -> TextCompletionRead:
    with Session(engine) as session:
        completion = session.get(TextCompletion, completion_id)
        if completion is None:
            raise HTTPException(status_code=404, detail="TextCompletion not found")
        return TextCompletionRead(id=completion.id, input=completion.input, output=completion.output)

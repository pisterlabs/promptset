from fastapi import APIRouter
from sqlalchemy import func, desc
import openai

import uuid
import datetime
from zoneinfo import ZoneInfo

from workspace.models import Workspace
from gpt_interactions.models import GptInteraction, FilledPrompt
from gpt_interactions.schemas import InteractionsResponse, InteractionSchema, GptRequestSchema, GptAnswerResponse
from init import sqlalchemy_session

router = APIRouter(prefix='/api',
                   tags=['GPT Interactions'])

def get_interactions(message: str) -> InteractionsResponse:
    with sqlalchemy_session.begin() as session:
        history = session.query(GptInteraction,
                                func.array_agg(FilledPrompt.text_data),
                                func.array_agg(FilledPrompt.number)) \
            .filter(GptInteraction.workspace_id == session.query(Workspace.id).filter(Workspace.initial).first()[0]) \
            .join(FilledPrompt).group_by(GptInteraction.id)\
            .order_by(desc(GptInteraction.time_happened))\
            .all()
        history = list(map(lambda el: InteractionSchema(
            id=el[0].id,
            request=GptRequestSchema(
                prompt=list(zip(*sorted(zip(el[1],el[2]), key=lambda el: el[1])))[0],
                username=el[0].username,
                company=el[0].company,
            ),
            datetime=el[0].time_happened,
            favorite=el[0].favorite,
            gpt_response=el[0].gpt_answer), history))
    return InteractionsResponse(status='success', message=message, data=history)


@router.put('/response')
def get_response(request: GptRequestSchema) -> GptAnswerResponse:
    response = openai.ChatCompletion.create(model='gpt-4', messages=[{'role': 'user', 'content': '\n'.join(request.prompt)}])
    answer = response['choices'][0]['message']['content']
    interaction_id = uuid.UUID(hex=str(uuid.uuid4()))
    with sqlalchemy_session.begin() as session:
        session.add(GptInteraction(id=interaction_id,
                                   gpt_answer=answer,
                                   username=request.username,
                                   favorite=False,
                                   company=request.company,
                                   time_happened=datetime.datetime.now(ZoneInfo('Europe/Moscow')),
                                   workspace_id=session.query(Workspace.id).filter(Workspace.initial).first()[0]))
        session.flush()
        session.add_all(map(lambda i, pr: FilledPrompt(id=uuid.UUID(hex=str(uuid.uuid4())),
                                                    text_data=pr,
                                                    gpt_interaction_id=interaction_id,
                                                    number=i),
                            *zip(*enumerate(request.prompt))))
    return GptAnswerResponse(status='success', message='GPT Response successfully retrieved', data={'gpt_response': answer})

@router.get('/history')
def get_history() -> InteractionsResponse:
    return get_interactions('History successfully retrieved')

@router.put('/favoriteHistory')
def add_to_favorite(id: uuid.UUID)->InteractionsResponse:
    with sqlalchemy_session.begin() as session:
        session.get(GptInteraction, id).favorite = True
    return get_interactions('Interaction successfully added to favorite')

@router.delete('/favoriteHistory')
def delete_from_favorite(id: uuid.UUID)->InteractionsResponse:
    with sqlalchemy_session.begin() as session:
        session.get(GptInteraction, id).favorite = False
    return get_interactions('Interaction successfully deleted from favorite')

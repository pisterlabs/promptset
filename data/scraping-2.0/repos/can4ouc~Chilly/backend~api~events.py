import ast
import os
import time
from datetime import datetime

import fastapi
from dotenv import load_dotenv
from fastapi import Depends, HTTPException
from openai import OpenAI
from starlette.responses import JSONResponse

from backend import db
from sqlalchemy.orm import Session

from backend.api.schema import EventSchema, EventInput, EventInfoForAI
from backend.models import Event, Tags

from fuzzywuzzy import fuzz

events_router = fastapi.APIRouter()
load_dotenv()
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
OPENAI_API_ORGANIZATION = os.environ['OPENAI_API_ORGANIZATION']

verification_codes = {}


@events_router.post('/events/get_ai_info', response_model=dict, status_code=201)
def create_event_step_one(input_info: EventInfoForAI) -> dict:
    response = dict()
    response['tags'] = list(get_tags_by_event(input_info.title, input_info.description))
    if not input_info.image:
        image = [generate_image_by_event(input_info.title, input_info.description)]
        response['image'] = image
    return response


@events_router.post('/events/create', response_model=EventSchema, status_code=201)
def create_event(creator_id: int, event: EventInput, db_session: Session = Depends(db.generate_session)):
    if not event.image:
        event.image = [generate_image_by_event(event.title, event.description)]
    db_event = Event(
        title=event.title,
        created_date=datetime.utcnow(),
        date=event.date,
        place=event.place,
        description=event.description,
        creator_id=creator_id,
        participants=[],
        tags=event.tags,
        image=event.image,
    )

    db_session.add(db_event)
    db_session.commit()
    db_session.refresh(db_event)
    return db_event


@events_router.get("/events/<int:event_id>", response_model=EventSchema, status_code=200)
def get_event_by_id(event_id: int, db_session: Session = Depends(db.generate_session)):
    event = db_session.query(Event).filter(Event.id == event_id).first()
    if event:
        return event

    raise HTTPException(
        status_code=404,
        detail='Event with requested id not found'
    )


@events_router.post("/events/<int:event_id>/attend", response_model=EventSchema, status_code=200)
def attend_event(event_id: int, user_id: int, db_session: Session = Depends(db.generate_session)):
    db_event = db_session.query(Event).filter_by(id=event_id).first()

    new_participants = [participant_id for participant_id in db_event.participants]
    if user_id in new_participants:
        raise HTTPException(
            status_code=404,
            detail='User already exist in this event'
        )

    new_participants.append(user_id)

    db_session.query(Event).filter(Event.id == event_id).update(
        {Event.participants: new_participants}, synchronize_session=False
    )
    db_session.flush()

    data = {"message": "OK"}
    return JSONResponse(content=data, status_code=200)


@events_router.get("/events/search/<str:query>", response_model=list[EventSchema], status_code=200)
def get_events_search_query(query: str, user_id, db_session: Session = Depends(db.generate_session)) -> list[Event]:
    # we need to retrieve user_id from current user
    events_list = db_session.query(Event).filter(
        Event.creator_id != user_id,
    ).all()
    # events_list = [event for event in events_list if user_id not in event.participants]
    match_scores = dict()
    events_dict = dict()
    for event in events_list:
        events_dict[event.id] = event
        match_scores[event.id] = calculate_event_query_match_score(query, event)
    # sort by match_score
    feed = dict(sorted(match_scores.items(), key=lambda item: item[1], reverse=True))
    return [events_dict[i] for i in feed.keys()]


def calculate_event_query_match_score(query: str, event: Event) -> float:
    title_match = fuzz.partial_ratio(query, event.title)/100
    description_match = fuzz.partial_ratio(query, event.description)/100
    tags_match = fuzz.partial_ratio(query, ' '.join(event.tags))/100
    location_match = fuzz.partial_ratio(query, event.location)/100
    return title_match * 0.5 + description_match * 0.3 + tags_match * 0.12 + location_match * 0.08


@events_router.get("/events/search_ai/<str:query>", response_model=list[EventSchema], status_code=200)
def get_events_search_ai(query: str, user_id, db_session: Session = Depends(db.generate_session)) -> list[Event]:
    # we need to retrieve user_id from current user
    events_list = db_session.query(Event).filter(
        Event.creator_id != user_id,
    ).all()
    events_list = [event for event in events_list if user_id not in event.participants]
    query_tags = get_tags_by_search_query(query)

    match_scores = dict()
    events_dict = dict()
    for event in events_list:
        events_dict[event.id] = event
        match_scores[event.id] = len(query_tags.intersection(set(event.tags)))
    # sort by match_score
    feed = dict(sorted(match_scores.items(), key=lambda item: item[1], reverse=True))
    return [events_dict[i] for i in feed.keys()]


def generate_image_by_event(title: str, description: str) -> str:
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        organization=OPENAI_API_ORGANIZATION,
    )

    response = client.images.generate(
        model="dall-e-3",
        prompt=(
            f"Please create a fun and catchy image using the colors #FC7310 and #F82B99 as main "
            f"and describing the following event: {title} – {description}"
        ),
        size="1024x1024",
        quality="standard",
        n=1,
    )

    return response.data[0].url


def get_tags_by_event(title: str, description: str) -> set:
    tags = [tag.value for tag in list(Tags)]

    client = OpenAI(
        api_key=OPENAI_API_KEY,
        organization=OPENAI_API_ORGANIZATION,
    )

    assistant = client.beta.assistants.create(
        name="Chilly AI search assistant",
        instructions=(
            "You're a personal assistant. "
            "I'll give you event title and description and you'll analyze it and generate relevant tags "
            f"matching the tags from this list: {tags}. Give me result as a python set."),
        model="gpt-3.5-turbo-1106"
    )

    thread = client.beta.threads.create()

    content = (
        f"Event title: {title}. Event description: {description}."
    )

    _message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=content,
    )

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )

    completed = False
    while not completed:
        time.sleep(2)
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
        if run.completed_at:
            completed = True

    messages = client.beta.threads.messages.list(
        thread_id=thread.id
    )

    return ast.literal_eval(messages.data[0].content[0].text.value)


def get_tags_by_search_query(query: str) -> set:
    tags = [tag.value for tag in list(Tags)]

    client = OpenAI(
        api_key=OPENAI_API_KEY,
        organization=OPENAI_API_ORGANIZATION,
    )

    assistant = client.beta.assistants.create(
        name="Chilly AI search assistant",
        instructions=(
            "You're a personal assistant. "
            "I'll give you search query of a person searching for events, "
            "and you'll analyze it and generate relevant tags "
            f"matching the tags from this list: {tags}. Give me result as a python set."),
        model="gpt-3.5-turbo-1106"
    )

    thread = client.beta.threads.create()

    content = (
        f"Search query: {query}."
    )

    _message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=content,
    )

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )

    completed = False
    while not completed:
        time.sleep(2)
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
        if run.completed_at:
            completed = True

    messages = client.beta.threads.messages.list(
        thread_id=thread.id
    )

    return ast.literal_eval(messages.data[0].content[0].text.value)


# if __name__ == '__main__':
#     print()
#     title = 'Freedom Candlemaker'
#     description = 'Freedom Candlemaker (Lefteris Moumtzis) is bringing his fiery new band from Greece to Sousami to perform his acclaimed new album, Beaming Light! The album was released in February 2019 on Inner Ear Records, with a premiere and a great review from famous UK online mag The 405. Many excellent reviews followed from Greek and international press. Τhe album was presented in Athens last April, and was met with amazing feedback from the Athenian audience.'
#     print(get_tags_by_event(title, description))
#     print(get_tags_by_event(title, description).intersection({'music', 'football'}))
    # print(generate_image(title, description))

# conftest.py
import datetime
import json
import os
from zoneinfo import ZoneInfo
from typing import List, Optional, Type

import pytest

from langchain.callbacks import get_openai_callback
from langchain.llms.fake import FakeListLLM

from langchain.agents.structured_chat.prompt import SUFFIX
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import faiss

from app import create_app
from bespokebots.dao.database import db


from bespokebots.services.google_calendar import (
    GoogleCalendarClient,
    GoogleCalendarEvent,
)

from bespokebots.services.chains.output_parsers import (
    CalendarAnalyzerOutputParserFactory,
    Thoughts,
    Analysis,
    Command,
    CalendarAnalysisResponse
)

from bespokebots.dao import User, ServiceProviders, CredentialStatus, OAuthStateToken
from bespokebots.dao.user_credentials import UserCredentials

from bespokebots.services.agent.todoist_tools import (CreateTaskTool, 
                                                      ViewProjectsTool,
                                                      CreateProjectTool)

from bespokebots.services.agent.bespoke_bot_agent import BespokeBotAgent
from bespokebots.services.chains.templates import (
    STRUCTURED_CHAT_PROMPT_PREFIX, 
    STRUCTURED_CHAT_PROMPT_SUFFIX
    )


@pytest.fixture(scope='session')
def app_fixture():
    """Session-wide test `Flask` application."""
    test_config = {
        'TESTING': True,
        'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:'
    }
    app = create_app(test_config=test_config)
    return app

@pytest.fixture(scope='session')
def _db(app_fixture):
    """Session-wide test database."""
    with app_fixture.app_context():
        db.create_all()
        yield db
        db.drop_all()

@pytest.fixture(scope='function')
def db_session(_db, app_fixture):
    with app_fixture.app_context():
        _db.create_all()
        yield _db.session
        _db.session.close()
        _db.drop_all()

@pytest.fixture(scope="function")
def test_user(db_session):
    user = User(username="test_user")
    db_session.add(user)
    db_session.commit()
    yield user

    db_session.delete(user)
    db_session.commit()




@pytest.fixture
def create_task_tool():
    tool = CreateTaskTool()
    yield tool

@pytest.fixture
def task(create_task_tool):
    task = create_task_tool.run({"content": "Test task"})
    yield task
    create_task_tool.todoist_client.delete_task(task['id'])

@pytest.fixture
def project_id():
    project_id = "2054677608"
    yield project_id

@pytest.fixture
def project_task(create_task_tool, project_id):
    task = create_task_tool.run({"content": "Test task", "project_id": project_id})
    yield task
    create_task_tool.todoist_client.delete_task(task['id'])

@pytest.fixture
def create_project_tool():
    tool = CreateProjectTool()
    yield tool

@pytest.fixture
def project_name():
    project_name = "Test Project"
    yield project_name

@pytest.fixture
def project(create_project_tool):
    project = create_project_tool.run({"name": "Test Project"})
    yield project
    create_project_tool.todoist_client.delete_project(project['id'])

@pytest.fixture
def child_project(create_project_tool,project):
    child = create_project_tool.run({"name": "Test Child Project", "parent_id": project['id']})
    yield child
    #project.create_project_tool.todoist_client.delete_project(child['id'])
    # I am pretty sure the parent project gets deleted after the yield in the project fixture.

@pytest.fixture
def lime_green_project(create_project_tool):
    project = create_project_tool.run({"name": "Test Project", "color": "lime_green"})
    yield project
    create_project_tool.todoist_client.delete_project(project['id'])



@pytest.fixture
def create_bespoke_bot_agent():
    def _create_bespoke_bot_agent():

        embeddings_model = OpenAIEmbeddings()
        # Initialize the vectorstore as empty
        
        embedding_size = 1536
        index = faiss.IndexFlatL2(embedding_size)
        vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

        bb_agent = BespokeBotAgent(
            ai_name="BespokeBot",
            memory=vectorstore
        )

        return bb_agent
    return _create_bespoke_bot_agent

@pytest.fixture
def generate_calendar_analysis_object():
    def _generate_calendar_analysis_object():
        thoughts = Thoughts(
            text = "figure out which day to schedule a thing",
            reasoning="I'm trying to figure out which day to schedule a thing",
            plan="I'm going to look at your calendar and see if I can find a day that works for you", 
            criticism="I'm assuming that you don't have any other commitments and that you're in the same timezone as your calendar events",
            speak="You are gonna need a bigger calendar"
        )
        analysis = Analysis(
            events_dates_comparison="Comparing the events' dates to 6/7 and 6/8.",
            event_conflict_detection="Checking for event conflicts on 6/7 and 6/8 at 9:30 AM EST for one hour.",
            available_time_slot_detection="Identifying available time slots on 6/7 and 6/8.",
            best_date_time_selection="Selecting the best date and time for the event.",
            answer="The best date and time for the event is 6/7 at 9:30 AM EST for one hour."
        )
        command = Command(
            name="schedule_event",
            args={
                "title": "A thing",
                "start": "2023-06-07T09:30:00-04:00",
                "end": "2023-06-07T10:30:00-04:00",
            }
        )
        return CalendarAnalysisResponse(
            thoughts=thoughts,
            analysis=analysis,
            command=command,
        )

    return _generate_calendar_analysis_object

@pytest.fixture
def events():
    def _events():
        return {
            "events": [
                {
                    "event_id": "6tukcd7qohtrijv8kic8limlvk_20230602",
                    "title": "Kids with me",
                    "start": "2023-05-29",
                    "end": "2023-06-08",
                },
                {
                    "event_id": "_88qkacho6ks3eba569238b9k8cpj8b9o6opk4ba388s4aga16sq3gghl6o",
                    "title": "Kel in South Dakota",
                    "start": "2023-05-30",
                    "end": "2023-06-06",
                },
                {
                    "event_id": "hsj31m6lrjd617m5b8meam59e8",
                    "start": "2023-05-31T10:00:00-04:00",
                    "end": "2023-05-31T11:00:00-04:00",
                    "description": "Appointment added to your calendar by Tom, your AI Assistant",
                    "summary": "Therapy w/ Alex",
                },
                {
                    "event_id": "5o7oodp4s58umohm8p6tdkb4i0",
                    "start": "2023-05-31T13:45:00-04:00",
                    "end": "2023-05-31T15:00:00-04:00",
                    "description": "Appointment added to your calendar by Tom, your AI Assistant",
                    "summary": "Physical exam w/ Dr. Curran",
                },
                {
                    "event_id": "55cuapjjkaqdp10a01obt88p2s",
                    "start": "2023-05-31T16:00:00-04:00",
                    "end": "2023-05-31T17:00:00-04:00",
                    "summary": "Absolute Chiro",
                },
                {
                    "event_id": "39tfjkkqoavueihtllsi6g8nto",
                    "start": "2023-05-31T17:00:00-04:00",
                    "end": "2023-05-31T17:30:00-04:00",
                    "description": "Pick up kids from school",
                    "summary": "Pick up kids",
                },
                {
                    "event_id": "ru89l98hleiq9gnppi3nl9eqd8",
                    "start": "2023-06-01T13:00:00-04:00",
                    "end": "2023-06-01T14:00:00-04:00",
                    "description": "Appointment with Dr. Hayner at SeaCoast Hand Therapy",
                    "summary": "SeaCoast Hand Therapy",
                    "location": "Scarborough, ME",
                },
                {
                    "event_id": "73ln88bvum4nstil015ql3ljkc",
                    "start": "2023-06-01T17:00:00-04:00",
                    "end": "2023-06-01T17:30:00-04:00",
                    "description": "Pick up kids from school",
                    "summary": "Pick up kids",
                    "location": "School",
                },
            ]
        }
    return _events

@pytest.fixture
def load_json_data():
    def _load_json_data(filename):
        with open(os.path.join("tests", "resources", filename)) as f:
            return json.load(f)

    return _load_json_data


@pytest.fixture
def create_future_event():
    def _create_future_event(
        calendar_id: str,
        summary: str,
        timezone: str,
        days_from_now: int,
        hours_long: int,
    ):
        """Create a future event on the calendar specified by the calendar_id.  The event will be
        days_from_now days from now and will be hours_long hours long.

        Args:
            calendar_id (str): The calendar id to create the event on
            summary (str): The title of the event
            timezone (str): The timezone to create the event in
            days_from_now (int): The number of days from now to create the event
            hours_long (int): The number of hours long the event is

        Returns:
            google_calendar_client (GoogleCalendarClient): The google calendar client
            response (dict): The response from the google calendar api
        """
        return create_calendar_event(calendar_id, summary, timezone, days_from_now, hours_long)

    return _create_future_event


@pytest.fixture
def create_event():
    def _create_event(
        calendar_id: str,
        summary: str,
        timezone: str
    ):
        return create_calendar_event(calendar_id, summary, timezone, 0, 1)

    return _create_event



def create_calendar_event(
    calendar_id: str,
    summary: str,
    timezone: str,
    days_from_now: Optional[int] = None,
    hours_long: Optional[int] = None,
):
    """Event creation helper function.  Assumes the event begins and ends on the same day.
    If the 'hours_long' parameter is not provided, the event will be 1 hour long.

    Args:   
        calendar_id (str): The calendar id to create the event on
        summary (str): The title of the event
        timezone (str): The timezone to create the event in
        days_from_now (int): The number of days from now to create the event
        hours_long (int): The number of hours long the event is

    Returns:
        google_calendar_client (GoogleCalendarClient): The google calendar client
        response (dict): The response from the google calendar api
    """
    tz = ZoneInfo(timezone)
    now = datetime.datetime.now(tz=tz).replace(microsecond=0)
    _hours_long = 1 or hours_long
    start_time = (now + datetime.timedelta(days=days_from_now)).isoformat()
    end_time = (
        now + datetime.timedelta(days=days_from_now, hours=_hours_long)
    ).isoformat()

    credentials = "../credentials.json"
    scopes = ["https://www.googleapis.com/auth/calendar"]

    google_calendar_client = GoogleCalendarClient(credentials, scopes)
    google_calendar_client.authenticate()

    event = GoogleCalendarEvent(tz, start_time, end_time, summary)
    event.description = "This event created by an automated test"
    response = google_calendar_client.create_event(calendar_id, event)

    return google_calendar_client, response






import pytest
from unittest.mock import patch
from langchain.llms.fake import FakeListLLM
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
#from bespokebots.services.chains.google_calendar_chain import GoogleCalendarChain
from tests.mocks.google_calendar_mock import GoogleCalendarMock

@pytest.fixture
def google_calendar_mock():
    return GoogleCalendarMock()

@pytest.fixture
def llm():
    responses=[
        "Action: Google Calendar: Retrieve Events\nAction Input: Retrieve events from 2023-06-01 to 2023-06-30",
        "Action: Google Calendar: Find Free Time\nAction Input: Find free time from 2023-06-01 to 2023-06-30",
        "Action: Google Calendar: Add Event\nAction Input: Add event to calendar with details 'Meeting with John' from 2023-06-15 10:00 to 2023-06-15 11:00",
        "Action: Google Calendar: Add Event\nAction Input: Add event to calendar with details 'Lunch with Sarah'",
        "Action: Google Calendar: Delete Event\nAction Input: Delete event 'Meeting with John'"
    ]
    return FakeListLLM(responses=responses)

@pytest.fixture
def agent(llm):
    tools = load_tools(["google_calendar"])
    return initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

@patch('my_project.google_calendar.GoogleCalendarAPI', new_callable=GoogleCalendarMock)
def test_retrieve_events(google_calendar_mock, agent):
    # Test here with agent and google_calendar_mock
    pass

@patch('my_project.google_calendar.GoogleCalendarAPI', new_callable=GoogleCalendarMock)
def test_retrieve_free_time(google_calendar_mock, agent):
    # Test here with agent and google_calendar_mock
    pass

@patch('my_project.google_calendar.GoogleCalendarAPI', new_callable=GoogleCalendarMock)
def test_add_event_with_date(google_calendar_mock, agent):
    # Test here with agent and google_calendar_mock
    pass

@patch('my_project.google_calendar.GoogleCalendarAPI', new_callable=GoogleCalendarMock)
def test_add_event_without_date(google_calendar_mock, agent):
    # Test here with agent and google_calendar_mock
    pass

@patch('my_project.google_calendar.GoogleCalendarAPI', new_callable=GoogleCalendarMock)
def test_delete_event(google_calendar_mock, agent):
    # Test here with agent and google_calendar_mock
    pass

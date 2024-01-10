import os
import json
import time
import pytest
import openai

from gpt_enterprise.employee import Employee
from gpt_enterprise.scrum_master import ScrumMaster
from gpt_enterprise.team_leader import TeamLeader


TEST_FILES_DIR = os.path.join(os.path.dirname(__file__), "test_files")
TASKS_FILES_DIR = os.path.join(TEST_FILES_DIR, "tasks")
EMPLOYEES_FILES_DIR = os.path.join(TEST_FILES_DIR, "employees")


# Change api url to LoacalAi one
openai.api_base = "local_ai_url"
openai.api_key = "sx-xxx"
OPENAI_API_KEY = "sx-xxx"
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

@pytest.fixture
def scrum_master_test():
    yield ScrumMaster(ceo_guidelines="Test", manager_retry=5, output_directory="Test")


@pytest.fixture
def team_leader_test():
    yield TeamLeader(ceo_guidelines="Test", manager_retry=5, output_directory="Test")


@pytest.fixture
def fake_employees():
    with open(os.path.join(EMPLOYEES_FILES_DIR, "employees.txt"), "r") as file:
        employees_to_hire = json.loads(file.read())
    hired_employees = {}
    for employee in employees_to_hire:
        hired_employees[employee["name"]] = Employee(
            role_prompt=employee["role"],
            name=employee["name"],
            role_name=employee["role_name"],
            creativity=float(employee["creativity"]),
            emoji=employee["emoji"],
        )
    return hired_employees


def mock_open_ai_response_object(mocker, content: str):
    """
    Mocks the response object from the openai api.
    """
    mock_generator_object = mocker.MagicMock()
    mock_message_object = mocker.MagicMock()
    mock_message_object.configure_mock(**{"message.content": content})
    mock_generator_object.configure_mock(**{"choices": [mock_message_object]})
    return mock_generator_object

import os

from assertpy import assert_that
from langchain.tools import Tool

from geospatial_agent.agent.geo_chat.chat_agent import GeoChatAgent
from geospatial_agent.agent.geo_chat.tools.geocode_tool import GEOCODE_TOOL
from geospatial_agent.agent.geo_chat.tools.gis_work_tool import GIS_WORK_TOOL
from geospatial_agent.shared.location import ENV_PLACE_INDEX_NAME

test_place_index = 'test_place_index'


def test_initializing_geo_chat_agent_does_not_raise_exception():
    os.environ[ENV_PLACE_INDEX_NAME] = test_place_index

    geo_chat_agent = GeoChatAgent()
    assert_that(geo_chat_agent).is_not_none()


def test_invoking_geo_chat_agent_does_not_raise_exception(mocker):
    mocker.patch(f'{GeoChatAgent.__module__}.geocode_tool',
                 return_value=Tool.from_function(func=lambda q: "geocoded response", name=GEOCODE_TOOL,
                                                 description="test description"))
    mocker.patch(f'{GeoChatAgent.__module__}.gis_work_tool',
                 return_value=Tool.from_function(func=lambda q: "gis work complete", name=GIS_WORK_TOOL,
                                                 description="test description"))

    mocker.patch(f'{GeoChatAgent.__module__}.AgentExecutor.run', return_value="The agent has finished running!")

    geo_chat_agent = GeoChatAgent()
    output = geo_chat_agent.invoke(
        agent_input="test input", session_id="test_session_id", storage_mode="test_storage_mode")

    assert_that(output).is_equal_to("The agent has finished running!")

from unittest.mock import Mock

from assertpy import assert_that
from langchain.tools import Tool

from geospatial_agent.agent.action_summarizer.action_summarizer import ActionSummarizer, ActionSummary
from geospatial_agent.agent.geo_chat.tools.gis_work_tool import gis_work_tool
from geospatial_agent.agent.geospatial.agent import GeospatialAgent


def test_initializing_gis_work_tool_does_not_raise_error():
    # Create a mock ActionSummarizer object
    mock_action_summarizer = Mock(spec=ActionSummarizer)
    # Create a mock GeospatialAgent object
    mock_geospatial_agent = Mock(spec=GeospatialAgent)

    tool = gis_work_tool(
        session_id='test-session-id',
        action_summarizer=mock_action_summarizer,
        gis_agent=mock_geospatial_agent,
        storage_mode='test-storage-mode'
    )

    assert_that(tool).is_not_none()
    assert_that(tool).is_instance_of(Tool)


def test_using_gis_work_tool_does_not_raise_error():
    mock_action_summarizer = ActionSummarizer
    mock_action_summarizer.invoke = Mock(
        return_value=ActionSummary(
            action="The user wants to draw a heatmap",
            file_summaries=[]
        ))

    mock_gis_agent = GeospatialAgent
    mock_gis_agent.invoke = Mock(return_value=None)

    tool = gis_work_tool(session_id='test_session_id',
                         action_summarizer=mock_action_summarizer,
                         gis_agent=mock_gis_agent,
                         storage_mode='test-storage-mode')

    output = tool.run(tool_input={
        'user_input': 'Draw me a heatmap!'
    })

    assert_that(output).is_equal_to("Observation: GIS Agent has completed it's work. This is the final answer.")

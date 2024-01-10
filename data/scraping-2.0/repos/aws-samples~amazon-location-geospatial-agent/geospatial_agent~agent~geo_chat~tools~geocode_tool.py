from langchain.tools import tool
from pydispatch import dispatcher

from geospatial_agent.shared.location import get_location_client, get_place_index_name

GEOCODE_TOOL = "geocode_tool"
GEOCODE_TOOL_FAILED = "geocode_tool_failed"


def geocode_tool(location_client=None, place_index_name: str = ""):
    if not location_client:
        location_client = get_location_client()

    if not place_index_name:
        place_index_name = get_place_index_name()

    @tool(GEOCODE_TOOL)
    def geocode_tool_func(query: str) -> str:
        """\
A tool that geocodes a given address using the AWS Location service.
The input is a string that could be an address, area, neighborhood, city, or country.
The output is list of places that match the input. Each place comes with a label and a
pair of coordinates following the format [longitude, latitude] that represents the physical location of the input.
        """

        try:
            response = location_client.search_place_index_for_text(
                IndexName=place_index_name,
                MaxResults=10,
                Text=query
            )

            response_string = ""
            for place in response['Results']:
                label_with_geom = f"{place['Place']['Label']}: {place['Place']['Geometry']['Point']}"
                response_string += label_with_geom + "\n"

            return response_string
        except Exception as e:
            dispatcher.send(signal=GEOCODE_TOOL_FAILED, sender=GEOCODE_TOOL, event_data=e)
            return "Observation: The tool did not find any results."

    return geocode_tool_func

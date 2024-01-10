from typing import Union, Any
from openai.types.shared_params import FunctionDefinition
from pydantic import ValidationError

updateMapTemplate: FunctionDefinition = {
    "name": "update_map",
    "description": "Update map to center on a particular location",
    "parameters": {
        "type": "object",
        "properties": {
            "longitude": {
                "type": "number",
                "description": "Longitude of the location to center the map on"
            },
            "latitude": {
                "type": "number",
                "description": "Latitude of the location to center the map on"
            },
            "zoom": {
                "type": "integer",
                "description": "Zoom level of the map"
            }
        },
        "required": ["longitude", "latitude", "zoom"]
    }
}

map_state: dict[str, Union[float, str]] = {
    "latitude": 39.949610,
    "longitude": -75.150282,
    "zoom": 16, }


# Function 1 - Coordinates to Control Map Location
def update_map(longitude: float, latitude: float, zoom: int):
    """Update map to center on a particular location and return status and map state."""

    if not longitude or not latitude or not zoom:
        return {"status": "Review Map Template", "map_state": updateMapTemplate}

    global map_state  # Refer to the global map_state

    try:
        # Update the map_state with validated data
        map_state['latitude'] = latitude
        map_state['longitude'] = longitude
        map_state['zoom'] = zoom

        # Return status and map_state in a dictionary
        return {"status": "Map Updated", "map_state": map_state}
    except (ValueError, TypeError) as e:
        # Return error status and message in a dictionary
        return {"status": f"Error in update_map function: {e}", "map_state": map_state}

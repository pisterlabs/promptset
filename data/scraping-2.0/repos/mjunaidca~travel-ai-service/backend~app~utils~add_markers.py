from typing import Union, Any
from openai.types.shared_params import FunctionDefinition

addMarkersTemplate: FunctionDefinition = {
    "name": "add_markers",
    "description": "Add list of markers to the map",
    "parameters": {
        "type": "object",
        "properties": {
            "longitudes": {
                "type": "array",
                "items": {
                    "type": "number"
                },
                "description": "List of longitude of the location to each marker"
            },
            "latitudes": {
                "type": "array",
                "items": {
                    "type": "number"
                },
                "description": "List of latitude of the location to each marker"
            },
            "labels": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "List of text to display on the location of each marker"
            }
        },
        "required": ["longitudes", "latitudes", "labels"]
    }
}


markers_state: dict[str, list[Any]] = {
    "latitudes": [],
    "longitudes": [],
    "labels": [],
}

# Function 2 - Add markers to map


def add_markers(latitudes: list[float], longitudes: list[float], labels: list[str]) -> dict[str, Union[str, Any]]:
    """Add list of markers to the map and return status and markers state."""

    if not longitudes or not latitudes or not labels:
        return {"status": "Review Add Markers Template", "markers_state": addMarkersTemplate}

    global map_state  # Refer to the global map_state

    try:
        markers_state["latitudes"] = latitudes
        markers_state["longitudes"] = longitudes
        markers_state["labels"] = labels

        return {"status": "Markers added successfully", "markers_state": markers_state}

    except (ValueError, TypeError) as e:
        return {"status": f"Error in add_markers function: {e}", "markers_state": markers_state}

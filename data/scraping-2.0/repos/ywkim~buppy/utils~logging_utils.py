from __future__ import annotations

import json
from typing import Any

from langchain.schema import BaseMessage


def custom_serializer(obj: object) -> str:
    """
    Custom serializer for complex objects.

    Args:
        obj (object): Object to serialize.

    Returns:
        str: Serialized string representation of the object.

    Raises:
        TypeError: If the object type cannot be serialized.
    """
    if isinstance(obj, BaseMessage):
        content = obj.content
        # Check if content is a list and contains base64 image data
        if isinstance(content, list):
            serialized_content: list[str | dict[Any, Any]] = []
            for item in content:
                if isinstance(item, dict) and item["type"] == "image_url":
                    # Shorten the base64 image data for logging
                    img_data = item["image_url"]["url"]
                    shortened_img_data = (
                        (img_data[:30] + "...") if len(img_data) > 30 else img_data
                    )
                    serialized_content.append(
                        {"type": "image_url", "image_url": {"url": shortened_img_data}}
                    )
                else:
                    serialized_content.append(item)
            return f"{obj.__class__.__name__}({serialized_content})"
        return f"{obj.__class__.__name__}({content})"
    if hasattr(obj, "__str__"):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def create_log_message(message: str, **kwargs) -> str:
    """
    Create a log message in JSON format.

    Args:
        message (str): Log message.
        **kwargs: Additional parameters for the log message.

    Returns:
        str: Log message in JSON format.
    """
    log_entry = {"message": message, **kwargs}
    return json.dumps(
        log_entry, default=custom_serializer, ensure_ascii=False, indent=4
    )

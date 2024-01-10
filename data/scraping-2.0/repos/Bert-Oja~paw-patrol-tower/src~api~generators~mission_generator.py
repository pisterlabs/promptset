"""
This module contains functions for generating missions.
"""
import os
import time
from typing import Any, Dict
import json
import logging
from openai import (
    APITimeoutError,
    RateLimitError,
    APIError,
)
from api.openai_integration import ChatApp
from api.prompts import mission_prompt, translation_prompt_1, translation_prompt_2


def generate_new_mission_data(
    previous_mission: Dict = None, chat_app: ChatApp = None
) -> Dict[str, Any]:
    """
    Generate new mission data.

    Args:
        previous_mission (Dict, optional): Previous mission data. Defaults to None.
        chat_app (ChatApp, optional): Chat application instance. Defaults to None.

    Returns:
        Dict[str, Any]: Generated mission data.
    """
    # Create a new chat app if none is provided
    if chat_app is None:
        chat_app = ChatApp(
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "2048")),
            top_p=1,
            frequency_penalty=0.3,
        )

    retry_delay = 5
    max_attempts = 3

    for _ in range(max_attempts):
        try:
            # Generate a new mission
            chat_app.set_system_message(mission_prompt)
            user_message = "Generate one mission"
            if previous_mission:
                user_message += (
                    f". Avoid location {previous_mission.get('main_location')}, "
                    f"pups {previous_mission.get('involved_pups')}, "
                    f"and title similar to \"{previous_mission.get('mission_title')}\""
                )

            mission_response = chat_app.chat(user_message)
            mission_response_data = json.loads(mission_response)

            # Translate the mission script
            chat_app.set_system_message(translation_prompt_1)
            chat_app.chat(mission_response_data.get("mission_script"))
            translation_response = json.loads(chat_app.chat(translation_prompt_2))

            mission_response_data["translation"] = translation_response.get(
                "translation"
            )
            return mission_response_data
        except (APITimeoutError, RateLimitError, APIError) as e:
            logging.warning(
                "%s encountered. Retrying after %s seconds.",
                type(e).__name__,
                retry_delay,
            )
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff
        except json.JSONDecodeError:
            logging.error("Failed to parse JSON response")
            return None

    logging.error("Max retry attempts reached. Unable to generate mission data.")
    return None

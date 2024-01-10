"""Calendar integration module."""
from datetime import datetime

import requests
from icalendar import Calendar

from utils.logger import get_logger
from data_retrieval.openai_integration import prompt_gpt4_turbo
from config.cfg import OPENAI_API_KEY, CAL_CONTEXT


def fetch_calendar_events(urls: list[str]) -> str:
    """
    Fetch calendar events from multiple .ics URLs and return a paragraph of natural prose describing the events.
    Only events for the current day are returned.

    :param urls: A list of .ics URLs.
    :return: A paragraph of natural prose describing the calendar events for the current day.
    """
    markdown_output = ""
    logger = get_logger()  # Initialize the logger
    today = datetime.now().date()  # Get the current date
    logger.info("Starting to fetch calendar events.")

    for url in urls:
        logger.info(f"Fetching calendar events from URL: {url}")
        try:
            response = requests.get(
                url, verify=False
            )  # Verify is set to False to ignore SSL certificate errors (temporary workaround)
            response.raise_for_status()

            calendar = Calendar.from_ical(response.content)
            logger.info(f"Successfully fetched and parsed calendar from {url}")

            for component in calendar.walk():
                if component.name == "VEVENT":
                    summary = component.get("summary")
                    start = component.get("dtstart")
                    end = component.get("dtend")

                    if start and start.dt:
                        # If start.dt is a datetime object, convert to date for comparison
                        start_date = (
                            start.dt.date()
                            if isinstance(start.dt, datetime)
                            else start.dt
                        )
                        if start_date == today:
                            start_fmt = (
                                start.dt.strftime("%Y-%m-%d %H:%M:%S")
                                if isinstance(start.dt, datetime)
                                else "All Day/Not Specified"
                            )
                            end_fmt = (
                                end.dt.strftime("%Y-%m-%d %H:%M:%S")
                                if end and isinstance(end.dt, datetime)
                                else "All Day/Not Specified"
                            )

                            markdown_output += f"- **{summary}**\n  - Start: {start_fmt}\n  - End: {end_fmt}\n\n"
        except requests.RequestException as e:
            logger.error(f"Failed to fetch calendar from {url}: {e}")
            return "..."
        except Exception as e:
            logger.error(f"Error: {e}")
            return "..."

    # If no events were found, return a message, otherwise return the markdown formatted string in natural language
    if not markdown_output:
        natural_language_output = "No events for today! :D"
    else:
        natural_language_output = prompt_gpt4_turbo(
            OPENAI_API_KEY, markdown_output, CAL_CONTEXT
        )

    logger.info("Finished fetching all calendar events.")
    return natural_language_output

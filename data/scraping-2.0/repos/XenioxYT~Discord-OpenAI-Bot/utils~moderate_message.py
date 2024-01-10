import openai
from dotenv import load_dotenv
import os
from datetime import datetime
import pytz
from utils.get_and_set_timezone import get_timezone, set_timezone

def moderate_content(message):
    """
    Moderates the content of the message and adjusts it according to the user's timezone.
    """

    # Load OpenAI API settings
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Fetch moderation results from OpenAI
    response = openai.Moderation.create(input=message.content)
    moderation_result = response["results"][0]

    # Fetch Discord ID and timezone
    discord_id = message.author.id
    user_timezone = get_timezone(discord_id)

    # Get current time in UTC and localize it to the user's timezone
    timestamp = datetime.utcnow()
    utc_time = pytz.utc.localize(timestamp)
    local_time = utc_time.astimezone(pytz.timezone(user_timezone))

    # Get the abbreviation of the current timezone (like BST, GMT, etc.)
    tz_abbr = local_time.tzname()

    # Format the time
    formatted_time = local_time.strftime("%Y-%m-%d %H:%M")

    if moderation_result["flagged"]:
        reasons = [category for category, flagged in moderation_result["categories"].items() if flagged]
        reason_string = ', '.join(reasons)
        print(f"Message flagged for: {reason_string} \nOriginal message: {message.content} by {message.author.display_name}")
        return f"{message.author.display_name} said: [content flagged for {reason_string}]"
    else:
        return f"At {formatted_time} {tz_abbr} {message.author.display_name} said: {message.content}"


# Example usage:
# class MockMessage:
#     def __init__(self, author_name, content):
#         self.author = MockAuthor(author_name)
#         self.content = content

# class MockAuthor:
#     def __init__(self, display_name):
#         self.display_name = display_name

# message = MockMessage("John", "[test]")
# result = moderate_content(message)
# print(result)

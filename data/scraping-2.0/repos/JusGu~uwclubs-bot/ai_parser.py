from openai import OpenAI
import mock_data
from consts.secrets import OPENAI_API_KEY

system_content = '''You are given a message on Discord.
You are to determine if it is an annoucement for an event. Reminder for deadlines are not events.

If it is an event, output JSON with the fields: title, start_time, end_time, description, and location.
Keep the title as concise as possible.
start_time and end_time are ISO 8601 strings and in EST unless otherwise specified.
If there is no end time, end_time is null.
Keep the description exactly as it is.
Location can be online or in a physical location or null.
If location is in a physical location, it should be in the format: "Building Name Room Number" without "room" in the middle.

If it is not an event, output JSON with the field: reason_for_error.
'''

import json

def parse_message(message: str) -> json:
    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": message}
            ]
        )
        responseJSON = json.loads(response.choices[0].message.content)
        if "reason_for_error" in responseJSON:
            return {"status": "error", "data": responseJSON, "original_message": message}
        else:
            return {"status": "success", "data": responseJSON}
    except Exception as e:
        return {"status": "unparseable", "data": {"reason_for_error": str(e)}, "original_message": message}

    
if __name__ == "__main__":
    parsed_message = parse_message(mock_data.message_2)
    print(json.dumps(parsed_message, indent=4))

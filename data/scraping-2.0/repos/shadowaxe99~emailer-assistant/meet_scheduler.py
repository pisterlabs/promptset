import openai
from uuid import uuid4

def query(role, content):
    return {"role": role, "content": content}

def create_event_format(suggested_time):
    # Dummy attendees for the meeting
    attendees = [{"email": "dummy_organizer@gmail.com"}, {"email": "dummy_guest1@gmail.com"}, {"email": "dummy_guest2@gmail.com"}]

    # For simplicity, let's assume the meeting duration is 1 hour. 
    # You can adjust this or extract the end time from the bot's response if needed.
    event_end_time = suggested_time + "T01:00:00"  # 1 hour after the start time

    event = {
        "summary": "test meeting",
        "start": {"dateTime": suggested_time},
        "end": {"dateTime": event_end_time},
        "attendees": attendees,
        "conferenceData": {
            "createRequest": {
                "requestId": f"{uuid4().hex}",
                "conferenceSolutionKey": {"type": "hangoutsMeet"}
            }
        },
        "reminders": {"useDefault": True}
    }
    return event

def schedule_meeting(reply, free_times):
    # Convert the list of free times into a string format for the context
    free_times_str = ", ".join(free_times)
    
    # Initial context for the bot, including the desired format
    context = (f"I need to decide if an online meeting is required based on the reply and suggest an appropriate time slot. "
               f"My available free times are: {free_times_str}. "
               f"Please respond in the format: 'meet_required = [Yes/No]\ntime = [suggested_time]' or just 'meet_required = No' if no meeting is needed.")
    
    # Create a list of messages to send to the bot
    messages_list = [
        query("system", context),
        query("user", reply)
    ]

    # Generate a response using GPT-4
    print("GENERATING RESPONSE")
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages_list
    )

    # Extract the bot's reply from the response
    bot_reply = response['choices'][0]['message']['content']

    # Check if a meeting is required
    if "meet_required = Yes" in bot_reply:
        # Extract the suggested time from the bot's reply
        suggested_time = bot_reply.split("time = ")[-1].strip()
        return create_event_format(suggested_time)
    else:
        return {}

# Example usage
if __name__ == "__main__":
    reply = "Let's discuss the project details further. How about tomorrow at 3pm?"
    free_times = ["Tomorrow from 2pm to 4pm", "Day after tomorrow from 10am to 12pm"]
    result = schedule_meeting(reply, free_times)
    print(result)

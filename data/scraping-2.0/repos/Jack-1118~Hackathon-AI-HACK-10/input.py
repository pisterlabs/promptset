from openai import OpenAI
import re
import json

# Initializing GPT Api
api_key = "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"
client = OpenAI(api_key=api_key)


# Generate the response
def answer(question):

    prompt = [{"role": "system", "content": "WanderBot is an advanced trip planner AI designed to crafting seamless itineraries."}, {"role": "user", "content": question}]
    response = client.chat.completions.create(
        model='ft:gpt-3.5-turbo-0613:personal::8WQpDXeX',
        messages=prompt 
        )
    
    response_msg = response.choices[0].message.content

    return response_msg


def insert_ai(country, state, day):
    command = f"Plan an itinerary based on following coondition, country: {country}, state: {state}, day: {day}"
    output = answer(command)
    # Split the input_string into days
    days = re.split(r';\s*', output)

    all_list = {}
    for day in days:
        # Extract the day number and events
        day_number, events_string = re.match(r'(Day \d+):(.*)', day).groups()
        
        # Find all events for the day
        events = re.findall(r'\{([^}]+)\}', events_string)

        for event in events:
            # Extract details for each event
            event_name = re.search(r'Event: ([^,]+)', event).group(1)
            place = re.search(r'Place: ([^,]+)', event).group(1)
            description = re.search(r'Remarks: ([^,]+)', event).group(1)
            link = re.search(r'Link: ([^\}]+)', event).group(1)

            # Add details to all_list
            if day_number not in all_list:
                all_list[day_number] = []
            all_list[day_number].append([event_name, place, description, link])

    json_string = json.dumps(all_list)
    with open('output.txt', 'w') as f:
        f.write(json_string)



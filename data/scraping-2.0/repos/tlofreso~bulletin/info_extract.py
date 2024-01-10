import json
from time import sleep
from typing import List, IO

from pydantic import BaseModel

from openai import Client

MASSTIME_PROMPT = """What are the regular Mass Times at this Parish?  Provide output as a valid JSON array in which every object in the array represents a single mass time.  Include attributes for the day of the week and the time of day.  The "day" attribute should be the name of the day, and the "time" attribute should be an int representing 24hr time.  (900 is 9am, 1400 is 2pm, etc.)

Example Response:

[
 {
    "day": "Sunday",
    "time": 900
 }
]

Do not include any content in the response other than the JSON itself.
"""

class MassTime(BaseModel):
    day: str  # "Monday"
    time: int # 1630 is 4:30pm. All times local

def get_mass_times(client:Client, assistant_id:str, bulletin_pdf:IO[bytes]) -> List[MassTime]:
    assistant = client.beta.assistants.retrieve(assistant_id)
    uploaded_bulletin=client.files.create(
        purpose="assistants",
        file=bulletin_pdf
    )

    thread = client.beta.threads.create()
    client.beta.threads.messages.create(
        thread_id=thread.id, 
        content=MASSTIME_PROMPT,
        role="user",
        file_ids=[uploaded_bulletin.id]
    )

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )

    while run.status in ["queued", "in_progress", "cancelling"]:
        #print(run.status)
        sleep(2)
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
    #print(run.status)

    messages = client.beta.threads.messages.list(
        thread_id=thread.id
    )

    client.files.delete(uploaded_bulletin.id)
    client.beta.threads.delete(thread.id)

    response_string = messages.data[0].content[0].text.value
    #print(response_string)
    json_str = response_string[response_string.find("[") : response_string.rfind("]") + 1]
    response_json = json.loads(json_str)
    response_masstimes = [MassTime.model_validate_json(json.dumps(j)) for j in response_json]
    return response_masstimes

if __name__ == '__main__':
    # Test code
    from tempfile import TemporaryFile
    from os import environ
    from download_bulletins import download_bulletin

    with TemporaryFile("w+b") as bulletin_file:
        download_bulletin("0689", bulletin_file)

        client = Client()
        assistant_id = environ["BULLETIN_ASSISTANT_ID"]

        mass_times = get_mass_times(client, assistant_id, bulletin_file)

        for mass_time in mass_times:
            print(mass_time)

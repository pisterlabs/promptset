import argparse
from pathlib import Path
from time import sleep
from typing import List

from pydantic import BaseModel

from openai import Client

class MassTime(BaseModel):
    day: str  # "Monday"
    time: int # 1630 is 4:30pm. All times local

def get_mass_times(client:Client, bulletin_pdf:Path) -> List[MassTime]:
    uploaded_bulletin=client.files.create(
        purpose="assistants",
        file=bulletin_pdf
    )

    assistant = client.beta.assistants.create(
        name="Masstime Extractor",
        instructions="You are an assistant for extracting information from Catholic Church bulletins. You read PDF files and can answer questions from them.",
        model="gpt-4-1106-preview",
        tools=[{"type": "retrieval"}]
    )

    thread = client.beta.threads.create()
    client.beta.threads.messages.create(
        thread_id=thread.id, 
        content="What are the regular Mass Times at this Parish?  Provide output as a valid JSON array in which every object in the array represents a single mass time.  Include attributes for the day of the week and the time of day.",
        role="user",
        file_ids=[uploaded_bulletin.id]
    )

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )

    while run.status in ["queued", "in_progress", "cancelling"]:
        print(run.status)
        sleep(5)
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
    print(run.status)

    messages = client.beta.threads.messages.list(
        thread_id=thread.id
    )

    print(messages)
    return []

def parse_args():
    parser = argparse.ArgumentParser(description='Extract Masstimes from a single file')
    parser.add_argument('pdf_path', type=str, help='The location of the PDF file.')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        raise IOError(f"File {pdf_path} does not exist")

    client = Client()

    mass_times = get_mass_times(client, Path(args.pdf_path))

    for mass_time in mass_times:
        print(mass_time)

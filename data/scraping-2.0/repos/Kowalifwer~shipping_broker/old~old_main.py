from fastapi import FastAPI, Depends, HTTPException, Query, Request, BackgroundTasks
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
from mail import EmailClientIMAP, EmailClientAzure
import asyncio
import configparser
from typing import Any, List, Optional, Union, Literal, Coroutine, Dict, Tuple, Callable
from datetime import datetime
from db import MongoEmail, MongoShip, MongoCargo
from pydantic import ValidationError
from gpt_prompts import prompt
import json
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from mail import EmailMessageAdapted
from fastapi.templating import Jinja2Templates
import openai
import mail_init
from contextlib import asynccontextmanager


# Imap details
# imap_server = "outlook.office365.com"
# imap_port = 993

# imap_client = EmailClientIMAP(imap_server, imap_port, email_address, email_password)
# imap_client.connect()

## All routes for FastAPI below

@app.get("/gpt")
async def gpt_prompt():
    example = """Iskenderun => 1 Italy Adriatic
Abt 4387 Cbm/937 mts Pipes
LENGHT OF PIPE: 20M
Total 83 pieces pipes
11 Nov/Onwards
4 ttl days sshex eiu
3.75%
+++

Batumi => Split 
5.000 mt urea in bb, sf'51 1000ex / 1500ex 
01-10 November 
3.75% 
+++
Saint Petersburg, Russia => Koper, Slovenia    
Abt 10'000 mts +10% in OO, SF abt 1.2 wog , 2 grades to be fully segregated 
15-20.11.2023
8000 mts SSHINC / 3500 mts SSHEX
EIU PWWD OF 24CONSECUTIVE HOURS  BENDS
CHABE
SDBC MAX 25 Y.O.
3.75%
+
"""

    response = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo-1106",
        # temperature=0.01,
        top_p=0.5,
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": example}
        ]
    )
    json_response = response.choices[0].message.content

    try:
        final = json.loads(json_response)
    except Exception as e:
        print("Error parsing JSON response from GPT-3", e)
        return {"message": "Error parsing JSON response from GPT-3", "original": json_response}

    return final


def process_email_dummy(email_message: EmailMessageAdapted) -> Union[bool, str]:
    return True

# Test view
@app.get("/delete_spam_emails_azure")
async def delete_spam_emails_azure():
    start_time = datetime.now()
    remaining_emails = await email_client.read_emails_and_delete_spam(500, unseen_only=False)
    print(f"Time taken to fetch emails and create objects, whilst launching background tasks: {datetime.now() - start_time}")

    if isinstance(remaining_emails, str):
        return {"error": remaining_emails}

    print(f"Number of remaining emails: {len(remaining_emails)}")

# Test view
@app.get("/list_mail_folders")
async def list_mail_folders():
    folders = await email_client.client.me.mail_folders.get()
    for folder in folders.value:
        messages = await email_client.client.me.mail_folders.by_mail_folder_id(folder.id).messages.get()
        print(f"{folder.display_name} has {len(messages.value)} messages")
        

    return {"message": folders}

async def email_to_entities_via_openai(email_message: EmailMessageAdapted) -> dict:
    
    #https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py -> parallel processing example
    """
    Convert an email message to JSON using OpenAI ChatCompletion.

    Parameters:
    - email_message (EmailMessageAdapted): The email message to be converted.

    Returns:
    Union[str, dict]: The JSON representation of the email message, or an error message.

    Raises:
    - OpenAIError: If there is an error in the OpenAI API request.
    - json.JSONDecodeError: If there is an error decoding the JSON response.
    """

    response = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo-1106",
        temperature=0.2,
        # top_p=1,
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": email_message.body}
        ]
    )
    json_response = response.choices[0].message.content # type: ignore

    final = json.loads(json_response)
    return final

async def insert_gpt_entries_into_db(entries: List[dict], email: MongoEmail) -> None:
    """Insert GPT-3.5 entries into database."""

    ignored_entries = []
    ships = []
    cargos = []

    for entry in entries:

        entry_type = entry.get("type")
        if entry_type not in ["ship", "cargo"]:
            ignored_entries.append(entry)
            continue

        entry["email"] = email

        if entry_type == "ship":
            try:
                ship = MongoShip(**entry)

                ships.append(ship.model_dump())
            except ValidationError as e:
                ignored_entries.append(entry)
                print("Error validating ship. skipping addition", e)
        
        elif entry_type == "cargo":
            try:
                cargo = MongoCargo(**entry)

                cargos.append(cargo.model_dump())
            except ValidationError as e:
                ignored_entries.append(entry)
                print("Error validating cargo. skipping addition", e)
    
    # Insert email into MongoDB
    await db["emails"].insert_one(email.model_dump())

    if ships:
        # Insert ships into MongoDB
        await db["ships"].insert_many(ships)

    if cargos:
        # Insert cargos into MongoDB
        await db["cargos"].insert_many(cargos)
    
    live_logger.report_to_channel("info", f"Inserted {len(ships)} ships and {len(cargos)} cargos into database.")
    if ignored_entries:
        live_logger.report_to_channel("warning", f"Additionally, ignored {len(ignored_entries)} entries from GPT-3.5. {ignored_entries}")

async def process_email(email_message: EmailMessageAdapted) -> None:

    email_added = await add_email_to_db(email_message)

    if not email_added:
        live_logger.report_to_channel("warning", f"Email with id {email_message.id} already in database. Ignoring.")
        return

    # Converting email to JSON via GPT-3.5
    try:
        gpt_response = await email_to_entities_via_openai(email_message)
    except Exception as e:
        live_logger.report_to_channel("error", f"Error converting email to JSON via GPT-3.5. {e}")
        return

    entries = gpt_response.get("entries", [])
    if not entries:
        live_logger.report_to_channel("error", f"Error in processing email - No entries returned from GPT-3.5.")
        return

    email: MongoEmail = email_message.mongo_db_object
    
    # For simplicity, the function below will handle logging to the live_logger
    await insert_gpt_entries_into_db(entries, email)

global_task_dict = {} # to track which endless tasks are running

async def endless_task(n: int):
    while global_task_dict.get(n, False): # while task is running. If task not initialized, return False
        print(f"Hello from the endless task {n}")
        await asyncio.sleep(3)

def start_endless_task(background_tasks: BackgroundTasks, n: int = 1):
    if not global_task_dict.get(n, False): # if task not initialized, initialize it
        global_task_dict[n] = True
        background_tasks.add_task(endless_task, n)

def stop_endless_task(n: int):
    global_task_dict[n] = False

@app.get("/start/{n}")
async def start_task(n: int, background_tasks: BackgroundTasks):
    start_endless_task(background_tasks, n)
    return {"message": f"Endless task {n} started."}

@app.get("/stop/{n}")
async def stop_tasks(n: int):
    stop_endless_task(n)
    return {"message": f"Endless task {n} stopped."}

@app.get("/stop_all")
async def stop_all_tasks():
    for key in global_task_dict:
        stop_endless_task(key)
    return {"message": "All tasks stopped."}

@app.get("/read_emails")
async def read_emails():
    emails = mail_handler.read_emails(
        #all emails search criteria
        search_criteria="ALL",
        num_emails=1,
        # search_keyword="MAP TA PHUT"
    )
    output = "no emails"
    if emails:
        print(emails[0].body)
        output = await process_email(emails[0])
        print(output)

        # for email_message in emails:
        #     # Process or display email details as needed
        #     db_entries.append(email_message.mongo_db_object.model_dump())
        
        # # Insert emails into MongoDB
        # await db["emails"].insert_many(db_entries)

        # #return html of last email
    return JSONResponse(content=output)

@app.get("/regex")
async def regex():
    import re
    cargo_email_text = """
    6912/232 mts
    8000/15000 PWWD SHINC BENDS
    END OCT/EARLY NOV 2023
    2.5% PUS
    """

    # Define the regex pattern
    pattern = re.compile(r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*mts?\b", re.IGNORECASE)

    # Find all matches in the text
    matches = pattern.findall(cargo_email_text)

    print(matches)

    return {"message": matches}

shutdown_background_processes = asyncio.Event()

# TODO: does not need to be async to be honest.
async def shutdown_handler():
    print("setting shutdown event")
    shutdown_background_processes.set()
    await live_logger.close_session()
    print("connection to websocket closed")

    # Shut off any running producer/consumer tasks
    for tasks in MQ_HANDLER.values():
        tasks[1].set()
        print(f"set shutdown event for task {tasks[0].__name__}")

    return {"message": "Shutdown event set"}

app.state.handle_shutdown = shutdown_handler

@app.get("/shutdown")
async def shutdown():
    print("setting shutdown event")
    await shutdown_handler()


if __name__ == "__main__":

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, use_colors=True)
from typing import Dict, Tuple, Callable, Coroutine, Any, Literal, List, Union, Optional
import asyncio
from openai import OpenAIError

from pydantic import ValidationError
from db import MongoShip, MongoCargo
from jinja2 import Template, Environment, FileSystemLoader
import json
from datetime import datetime, timedelta
from bson import ObjectId

from setup import email_client, openai_client, db_client
from realtime_status_logger import live_logger
from mail import EmailMessageAdapted
from db import MongoEmail, update_ship_entry_with_calculated_fields, update_cargo_entry_with_calculated_fields
from mq import scoring
from gpt_prompts import prompt

async def mailbox_read_producer(stoppage_event: asyncio.Event, queue: asyncio.Queue[EmailMessageAdapted]):
    live_logger.report_to_channel("extra", f"Starting MULTI mailbox read producer.")
    asyncio.create_task(_mailbox_read_producer(stoppage_event, queue, True))
    # asyncio.create_task(_mailbox_read_producer(stoppage_event, queue, False))

async def _mailbox_read_producer(stoppage_event: asyncio.Event, queue: asyncio.Queue[EmailMessageAdapted], reverse=True):
    # 1. Read all UNSEEN emails from the mailbox, every 5 seconds, or if email queue is processed.

    attempt_interval = 5 # seconds

    while not stoppage_event.is_set():

        email_generator = email_client.endless_email_read_generator(
            n=9999,
            batch_size=50,

            most_recent_first=reverse,
            unseen_only=True,
            set_to_read=False,
            remove_undelivered=True,
        )

        async for email_batch in email_generator:
            batch_processed = False # This flag will indicate when it is time to fetch the next batch of emails

            while not batch_processed: # Will continuosuly try to add emails to the queue, until the batch is processed.

                if isinstance(email_batch, str):
                    live_logger.report_to_channel("error", f"Error reading emails from mailbox. Closing producer.")
                    break

                if isinstance(email_batch, list) and not email_batch:
                    live_logger.report_to_channel("info", f"No emails found in mailbox.")
                    break
                
                for count, email in enumerate(email_batch, start=1):

                    while True: # Will continuously try to add emails to the queue, until the queue has space. Make sure to break out of this loop when the email is added to the queue.
                        try:
                            if stoppage_event.is_set(): # if producer got stopped in the middle of processing emails, then we must unset the read flag for the remaining emails
                                ##for remaining emails, unset the read flag
                                # email_ids = [email.id] + [email.id for email in emails[count:]]
                                # asyncio.create_task(email_client.set_email_seen_status(email_ids, False))
                                live_logger.report_to_channel("warning", f"Failed to add whole email batch to queue. Producer closing before more space free'd up. Cleanup in progress.")
                                live_logger.report_to_channel("info", f"Producer closed verified.")
                                return

                            queue.put_nowait(email)
                            live_logger.report_to_channel("info", f"Email {count} placed in queue.")
                            break # break out of the while loop

                        except asyncio.QueueFull:
                            live_logger.report_to_channel("warning", f"{queue.qsize()}/{queue.maxsize} emails in messenger queue - waiting for queue to free up space.")
                            await asyncio.sleep(attempt_interval)

                # If full email batch was added to queue, then break out of the while loop
                if stoppage_event.is_set():
                    live_logger.report_to_channel("warning", f"producer closed verified (after adding a full batch of emails to queue)")
                    return

                live_logger.report_to_channel("info", f"Full email batch added to MQ succesfully.")
                batch_processed = True

                await asyncio.sleep(0.2)
        
        # If we are here - that means the generator has been exhaused! meaning all emails have been read OR n limit has been reached.
        # Therefore, it would be a good idea to wait a bit before starting a new cycle.
        live_logger.report_to_channel("info", f"Email generator exhausted. Waiting 10 seconds before starting a new cycle.")
        await asyncio.sleep(10)

    live_logger.report_to_channel("info", f"Producer closed verified.")

async def mailbox_read_consumer(stoppage_event: asyncio.Event, queue_to_fetch_from: asyncio.Queue[EmailMessageAdapted], queue_to_add_to: asyncio.Queue[EmailMessageAdapted], default=True):
    # 2. Process all emails in the queue, every 10 seconds, or if email queue is processed.
    
    # create a fake queue, by pulling most recent emails from the database

    while not stoppage_event.is_set():
        try:
            email = queue_to_fetch_from.get_nowait() # get email from queue
        except asyncio.QueueEmpty:
            await asyncio.sleep(1)
            continue

        email_added = await add_email_to_db(email)
        if email_added == False:
            continue

        # TODO: This currently also acts as a producer (until it is decided what the final approach will be) which can cause bottlenecks. 
        # Refer to: https://github.com/Kowalifwer/shipping_broker/issues/2
        while True:
            try:
                queue_to_add_to.put_nowait(email)
                live_logger.report_to_channel("info", f"Email with id {email.id} added to queue.")
                break # break out of the while loop
            except asyncio.QueueFull:
                live_logger.report_to_channel("warning", f"{queue_to_add_to.qsize()}/{queue_to_add_to.maxsize} email id's in messenger queue - waiting for queue to free up space.")
                await asyncio.sleep(5)
                continue

    live_logger.report_to_channel("info", f"Consumer closed verified.")

async def gpt_email_consumer(stoppage_event: asyncio.Event, queue: asyncio.Queue[EmailMessageAdapted], n_tasks: int = 1):
    live_logger.report_to_channel("gpt", f"Summoned {n_tasks} GPT-3 email consumers.")
    # Summon n consumers to run concurrently, and turn emails from queue into entities using GPT-3 (THIS IS ALMOST FULLY A I/O BOUND TASK, so should not be too CPU intensive)

    semaphore = asyncio.Semaphore(n_tasks) # limit the number of concurrent tasks to n_tasks * 10
    while not stoppage_event.is_set():
        tasks = [asyncio.create_task(_gpt_email_consumer(stoppage_event, queue, semaphore)) for _ in range(n_tasks * 10)]

        await asyncio.gather(*tasks, return_exceptions=True)


    live_logger.report_to_channel("gpt", f"Consumer closed verified.")

async def _gpt_email_consumer(stoppage_event: asyncio.Event, queue: asyncio.Queue[EmailMessageAdapted], semaphore: asyncio.Semaphore):
    async with semaphore:
        # 5. Consume emails from the queue, and generate a response using GPT-3
        if stoppage_event.is_set():
            return

        try:
            email = queue.get_nowait() # get email from queue
            gpt_response = await email_to_entities_via_openai(email)

            entries = gpt_response.get("entries", [])
            if not entries:
                live_logger.report_to_channel("error", f"Error in processing email - No entries returned from GPT-3.5.")
                return
            
            await insert_gpt_entries_into_db(entries, email)

            live_logger.report_to_channel("gpt", f"Email with id {email.id} processed by GPT-3.5. Entities added to database. Sleeping for 5 seconds.")
            await asyncio.sleep(2)

        except asyncio.QueueEmpty:
            await asyncio.sleep(2)

        except OpenAIError as e:
            live_logger.report_to_channel("gpt", f"Error converting email to entities via OpenAI. {e}")
        
        except json.JSONDecodeError as e:
            live_logger.report_to_channel("gpt", f"Error decoding JSON response from OpenAI. {e}")
        
        except Exception as e:
            live_logger.report_to_channel("gpt", f"Unhandled error in processing email. {e}")

async def item_matching_producer(stoppage_event: asyncio.Event, queue: asyncio.Queue[MongoShip]):
    while not stoppage_event.is_set():

        # Fetch emails from DB, about SHIPS, from most RECENT to least recent, and add them to the queue
        # Make sure the ships "pairs_with" is an empty list, and that the ship has not been processed yet.
        # From past 3 days
        date_from = datetime.utcnow() - timedelta(days=3)
        db_cursor = db_client["ships"].find(
            {
                "pairs_with": [],
                "timestamp_created": {"$gte": date_from}
            }
        ).sort("timestamp_created", -1)

        async for ship in db_cursor:
            ship = MongoShip(**ship)
            await queue.put(ship)
            if stoppage_event.is_set():
                live_logger.report_to_channel("info", f"Producer closed verified.")
                return

async def item_matching_consumer(stoppage_event: asyncio.Event, queue_from: asyncio.Queue[MongoShip], queue_to: asyncio.Queue[MongoShip]):
    # 6. Consume emails from the queue, and match them with other entities in the database
    while not stoppage_event.is_set():
        try:
            ship = queue_from.get_nowait() # get ship from queue

            matching_cargos = await match_cargos_to_ship(ship)

            if not matching_cargos:
                live_logger.report_to_channel("warning", f"No matching cargos found for ship with id {str(ship.id)}.")
                continue
            
            filename = "example_matches.json"
            try:
                with open(filename, "r") as file:
                    existing_data = json.load(file)
            except FileNotFoundError:
                existing_data = []
            

            existing_data.append({
                    "ship": ship.name,
                    "ship_port": ship.port,
                    "ship_sea": ship.sea,
                    "ship_month": ship.month,
                    "ship_email_contents": ship.email.body,
                    "ship_quantity": ship.capacity_int,
                    "matching_cargos": matching_cargos
                })

            with open(filename, "w") as file:
                json.dump(existing_data, file, indent=4)
                
            ship.pairs_with = [ObjectId(cargo["id"]) for cargo in matching_cargos]

            # Update ship in database
            # res = await db_client["ships"].update_one({"_id": ship.id}, {"$set": {"pairs_with": ship.pairs_with, "timestamp_pairs_updated": datetime.now()}})
            # if not res.acknowledged:
            #     live_logger.report_to_channel("error", f"Error updating ship with id {str(ship.id)} in database.")
            #     continue

            while True:
                try:
                    queue_to.put_nowait(ship)
                    break
                except asyncio.QueueFull:
                    if stoppage_event.is_set():
                        live_logger.report_to_channel("info", f"Consumer closed verified.")
                        return

                    live_logger.report_to_channel("warning", f"{queue_to.qsize()}/{queue_to.maxsize} ships in matching queue - waiting for queue to free up space.")
                    await asyncio.sleep(5)
        
        except asyncio.QueueEmpty:
            await asyncio.sleep(2)
    
    live_logger.report_to_channel("info", f"Consumer closed verified.")

async def email_send_producer(stoppage_event: asyncio.Event, queue: asyncio.Queue[MongoShip]):
    # 7. Consume emails from the queue, and send emails to the relevant recipients
    while not stoppage_event.is_set():
        try:
            ship = queue.get_nowait() # get ship from queue
            if not ship.pairs_with:
                live_logger.report_to_channel("warning", f"Ship with id {str(ship.id)} has no matching cargos. CRITICAL ERROR SHOULD NOT HAPPEN!!")
                continue

            # At this point, we have a MongoShip object, with a list of id's of matching cargos.
            # That is enough to send our emails.

            # Fetch the cargoes from the database
            db_cargos = await db_client["cargos"].find({"_id": {"$in": ship.pairs_with}}).to_list(None)
            cargos = [MongoCargo(**cargo) for cargo in db_cargos]

            body = render_email_body_text({
                "cargos": cargos,
                "ship": ship,
                "email": ship.email,
            })

            success = await email_client.send_email(
                to_email="shipperinho123@gmail.com",
                subject="Cargo Matching TEST",
                body=body,
            )

            if not success:
                live_logger.report_to_channel("error", f"Error sending email to {ship.email.sender}.")
                continue
        
        except asyncio.QueueEmpty:
            await asyncio.sleep(2)
    
    live_logger.report_to_channel("info", f"Producer closed verified.")

async def queue_capacity_producer(stoppage_event: asyncio.Event, *queues: asyncio.Queue):
    # 3. Check the queue capacity every 5 seconds, and report to the live logger
    from uuid import uuid4
    q_ids = [str(uuid4()) for _ in queues]

    while not stoppage_event.is_set():
        for i, queue in enumerate(queues):
            live_logger.report_to_channel("capacities", f"{queue.qsize()},{queue.maxsize},{q_ids[i]}", False)

        await asyncio.sleep(0.2)
    
    live_logger.report_to_channel("info", f"Queue capacity producer closed verified.")

async def flush_queue(stoppage_event: asyncio.Event, queue: asyncio.Queue):

    ##remove all items from queue
    while not queue.empty() and not stoppage_event.is_set():
        queue.get_nowait()
        await asyncio.sleep(0.1)
    
    live_logger.report_to_channel("info", f"Queue flushed.")

async def endless_trash_email_cleaner(stoppage_event: asyncio.Event):
    """This function will run until told to stop, scanning the mailbox for emails and deleting all the trash."""
    batch_size = 50

    while not stoppage_event.is_set():
        async for emails in email_client.endless_email_read_generator(n=99999, batch_size=batch_size, unseen_only=False, most_recent_first=True, remove_undelivered=True, set_to_read=False):
            live_logger.report_to_channel("trash_emails", f"{len(emails)} emails processed. Deleting {batch_size - len(emails)}/{batch_size} this round.")

            # If process stopped - break out of the infinite generator
            if stoppage_event.is_set():
                break

        # If the process finished on its own (i.e all the emails have been read OR n limit has been reached), then wait 20 seconds before starting a new cycle.
        else:
            live_logger.report_to_channel("trash_emails", f"Email generator exhausted. Waiting 20 seconds before starting a new cycle.")
            await asyncio.sleep(20)

    live_logger.report_to_channel("info", f"Consumer closed verified.")

# Message Queue for stage 1 - Mailbox read and add to database
MQ_MAILBOX: asyncio.Queue[EmailMessageAdapted] = asyncio.Queue(maxsize=2000)

# Message Queue for stage 2 - GPT-3.5 email processing
MQ_GPT_EMAIL_TO_DB: asyncio.Queue[EmailMessageAdapted] = asyncio.Queue(maxsize=500)

# Message Queue for stage 3 - Item matching
MQ_ITEM_MATCHING: asyncio.Queue[MongoShip] = asyncio.Queue(maxsize=5)

# Message Queue for stage 4 - Email send for Ships with matched cargos
MQ_EMAIL_SEND: asyncio.Queue[MongoShip] = asyncio.Queue(maxsize=1)

# Please respect the complex signature of this dictionary. You have to create your async functions with the specified signature, and add them to the dictionary below.
MQ_HANDLER: Dict[str, Tuple[
        Callable[[asyncio.Event, asyncio.Queue], Coroutine[Any, Any, None]], 
        asyncio.Event,
        asyncio.Queue]
    ] = {
    "mailbox_read_producer": (mailbox_read_producer, asyncio.Event(), MQ_MAILBOX),   
    "mailbox_read_consumer": (mailbox_read_consumer, asyncio.Event(), MQ_MAILBOX, MQ_GPT_EMAIL_TO_DB), # type: ignore - temporary due to https://github.com/Kowalifwer/shipping_broker/issues/2


    "5_gpt_email_consumer": (gpt_email_consumer, asyncio.Event(), MQ_GPT_EMAIL_TO_DB), #temporarily, declare number of tasks in the function name

    "item_matching_producer": (item_matching_producer, asyncio.Event(), MQ_ITEM_MATCHING),
    "item_matching_consumer": (item_matching_consumer, asyncio.Event(), MQ_ITEM_MATCHING, MQ_EMAIL_SEND), # type: ignore - temporary due to

    "email_send_producer": (email_send_producer, asyncio.Event(), MQ_EMAIL_SEND), # type: ignore - temporary due to "item_matching_consumer


    # "db_listens_for_new_emails_producer": (db_listens_for_new_emails_producer, asyncio.Event(), MQ_GPT_EMAIL_TO_DB),
    # "gpt_email_producer": (gpt_email_producer, asyncio.Event(), MQ_GPT_EMAIL_TO_DB),


    # temporary helper methods for testing.
    "queue_capacity_producer": (queue_capacity_producer, asyncio.Event(), MQ_MAILBOX, MQ_GPT_EMAIL_TO_DB, MQ_ITEM_MATCHING, MQ_EMAIL_SEND),
    "endless_trash_email_cleaner_producer": (endless_trash_email_cleaner, asyncio.Event()),
    # "flush_queue_producer": (flush_queue, asyncio.Event(), MQ_MAILBOX),
}
"""
This dictionary (MQ_HANDLER) stores the callables for different message queue handlers, along with the event and queue objects that they will be using.
The functionality should be handled by adding new methods in process_manager.py, and then adding them to this dictionary, following the correct format.
"""

def setup_to_frontend_template_data() -> List[Dict[str, str]]:
    """A helper function to convert the MQ_HANDLER dictionary into a format that can be used by the frontend template."""

    buttons = []
    for key in MQ_HANDLER:
        title_words = key.split("_")
        task_type = title_words[-1]
        title = " ".join(title_words).capitalize()

        buttons.append({
            "name": title,
            "start_url": f"/start/{task_type}/{key}",
            "end_url": f"/end/{task_type}/{key}"
        })
    return buttons

async def add_email_to_db(email_message: EmailMessageAdapted) -> Union[str, bool]:
    """Add email to database, if it doesn't already exist. Return True if added, False if already exists."""

    try:
        email_in_db = await db_client["emails"].find_one({
            "$or": [
                {"id": email_message.id},
                {"body": email_message.body}
                # {"subject": email_message.subject, "sender": email_message.sender},
            ]
        })
    except Exception as e:
        live_logger.report_to_channel("error", f"Error finding email in database. {e}")
        return False

    if email_in_db:
        # TODO: consider updating the fields on the duplicate object, such as date_recieved or store a counter of duplicates, if this into will be useful later.
        return False

    mongo_email = email_message.mongo_db_object
    mongo_email.timestamp_added_to_db = datetime.now()

    inserted_result = await db_client["emails"].insert_one(mongo_email.model_dump())

    if not inserted_result.acknowledged:
        live_logger.report_to_channel("error", f"Error adding email to database.")
        return False

    live_logger.report_to_channel("info", f"Email with id {email_message.id} added to database.")

    return str(inserted_result.inserted_id)

async def email_to_entities_via_openai(email_message: EmailMessageAdapted) -> dict:
    
    #https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py -> parallel processing example
    """
    Convert an email message to JSON using OpenAI ChatCompletion.

    Parameters:
    - email_message: The email message to convert to JSON. An instance of EmailMessageAdapted.

    Returns:
    Union[str, dict]: The JSON representation of the email message, or an error message.

    Raises:
    - OpenAIError: If there is an error in the OpenAI API request.
    - json.JSONDecodeError: If there is an error decoding the JSON response.
    """

    response = await openai_client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        temperature=0.2,
        # top_p=1,
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": email_message.body} # type: ignore
        ]
    )
    json_response = response.choices[0].message.content # type: ignore
    if not json_response:
        raise OpenAIError("No JSON response from OpenAI.")

    final = json.loads(json_response)
    return final

async def insert_gpt_entries_into_db(entries: List[dict], email: EmailMessageAdapted) -> None:
    """Insert GPT-3.5 entries into database."""

    ignored_entries = []
    ships = []
    cargos = []
    mongo_email = email.mongo_db_object

    for entry in entries:

        entry_type = entry.pop("type", None)
        if entry_type not in ["ship", "cargo"]:
            ignored_entries.append(entry)
            continue

        entry["email"] = mongo_email

        if entry_type == "ship":
            try:
                # Add calculated fields to ship
                update_ship_entry_with_calculated_fields(entry)

                ship = MongoShip(**entry)

                ships.append(ship.model_dump())
            except ValidationError as e:
                ignored_entries.append(entry)
                print("Error validating ship. skipping addition", e)
        
        elif entry_type == "cargo":
            try:
                # Add calculated fields to cargo
                update_cargo_entry_with_calculated_fields(entry)

                cargo = MongoCargo(**entry)

                cargos.append(cargo.model_dump())
            except ValidationError as e:
                ignored_entries.append(entry)
                print("Error validating cargo. skipping addition", e)

    if ships:
        # Insert ships into MongoDB
        ships_inserted_ids = await db_client["ships"].insert_many(ships)
        ship_ids = [str(id) for id in ships_inserted_ids.inserted_ids if id]
        # Update email object with ship ids
        mongo_email.extracted_ship_ids = ship_ids
        # Send update to the database
        res = await db_client["emails"].update_one({"_id": mongo_email.m_id}, {"$set": {"extracted_ship_ids": ship_ids}})

    if cargos:
        # Insert cargos into MongoDB
        cargos_inserted_ids = await db_client["cargos"].insert_many(cargos)
        cargo_ids = [str(id) for id in cargos_inserted_ids.inserted_ids if id]
        mongo_email.extracted_cargo_ids = cargo_ids
        # Send update to the database
        await db_client["emails"].update_one({"_id": mongo_email.m_id}, {"$set": {"extracted_cargo_ids": cargo_ids}})
    
    # Update emails timestamp_entities_extracted to now
    await db_client["emails"].update_one({"id": mongo_email.id}, {"$set": {"timestamp_entities_extracted": datetime.now()}})
    
    live_logger.report_to_channel("gpt", f"Inserted {len(ships)} ships and {len(cargos)} cargos into database.")
    if ignored_entries:
        live_logger.report_to_channel("extra", f"Additionally, ignored {len(ignored_entries)} entries from GPT-3.5. {ignored_entries}")
    
async def match_cargos_to_ship(ship: MongoShip, max_n: int = 5) -> List[Any]:
    """Match cargos to a ship, based on the extracted fields."""

    # Try to consider the following:
    # 1. capacity_int (if specified) - both ship and cargo have a value. make it a match IF capacities are within 10% of each other.
    # 2. month_int (if specified) - both ship and cargo have a value. make it a match IF months are the same or within 1 month of each other.
    # 3. port (if specified) - both ship and cargo have a value. cargo has port_from port_to, whilst ship only has port. make it a match IF port is the same as either port_from or port_to.
    # 4. sea (if specified) - both ship and cargo have a value. cargo has sea_from sea_to, whilst ship only has sea. make it a match IF sea is the same as either sea_from or sea_to.
    
    # Ship quantity_min_int and quantity_max_int.
    # Ship month_int
    # Ship port_to, port_from
    # Ship sea_to, sea_from

    # STAGE 1 - HARD FILTERS (DB QUERY) - stuff that will completely disqualify a cargo from being matched with a ship (can be done fully via DB query)
    # TBD... (for now we retrieve all cargos, since we don't have enough data to filter them out)

    #fetch cargos from past 3 days only
    date_from = datetime.now() - timedelta(days=3)
    db_cargos = await db_client["cargos"].find({
        # "timestamp_created": {"$gte": date_from}
    }).sort(
        "timestamp_created", -1
    ).to_list(None)
    cargos = [MongoCargo(**cargo) for cargo in db_cargos]
    simple_scores = []

    port_embeddings = []
    sea_embeddings = []
    general_embeddings = []
    
    for cargo in cargos:
        port_embeddings.append(cargo.port_embedding)
        sea_embeddings.append(cargo.sea_embedding)
        general_embeddings.append(cargo.general_embedding)

        score = 0
        # STAGE 2 - BASIC DB FIELDS SCORE -> CALCULATE SCORE from simple fields, such as capacity_int, month_int, comission_float...

        # 1. Handle Ship Capacity vs Cargo Quantity logic
        score += scoring.capacity_modifier(ship, cargo)

        # 2. Handle Ship Month vs Cargo Month logic
        score += scoring.month_modifier(ship, cargo)

        # 3. Handle Cargo comission scoring
        score += scoring.comission_modifier(ship, cargo)

        # 4. Handle date created scoring
        # score += scoring.timestamp_created_modifier(ship, cargo)

        simple_scores.append(score)

    # STAGE 3 - EMBEDDINGS SCORE -> CALCULATE SCORE from embeddings, such as port_embedding, sea_embedding, general_embedding...
    from sklearn.metrics.pairwise import cosine_similarity

    # Normalize simple scores based on max and min values in the list

    simple_scores = scoring.min_max_scale_robust(simple_scores, -0.1, 1) # Normalized to be between -0.1 and 1.0

    # simple_scores
    sea_scores = cosine_similarity([ship.sea_embedding], sea_embeddings)[0]
    port_scores = cosine_similarity([ship.port_embedding], port_embeddings)[0]
    general_scores = cosine_similarity([ship.general_embedding], general_embeddings)[0]

    # STAGE 4 - FINAL SCORE -> COMBINE THE SCORES FROM STAGE 2 AND STAGE 3, AND SORT THE CARGOS BY SCORE
    # Mean rank? Weighted Sum (normalize scores first)? TBD...
    final_scores = [
        {
            "id": str(cargos[i].id),
            "cargo_capacity_max": cargos[i].quantity_max_int,
            "cargo_capacity_min": cargos[i].quantity_min_int,
            "cargo_month": cargos[i].month,
            "cargo_port_from": cargos[i].port_from,
            "cargo_port_to": cargos[i].port_to,
            "cargo_sea_from": cargos[i].sea_from,
            "cargo_sea_to": cargos[i].sea_to,
            "cargo_commission": cargos[i].commission,

            "total_score": sum(scores),
            "simple_score": simple_scores[i],
            "port_score": port_scores[i],
            "sea_score": sea_scores[i],
            "general_score": general_scores[i],
            "email_body": cargos[i].email.body,
        } for i, scores in enumerate(zip(
            simple_scores, # Simple score (db scores) are important - hence should be multiplied
            port_scores * 1.25,
            sea_scores,
            general_scores * 1.75
        )
    )]
    final_scores.sort(key=lambda x: x["total_score"], reverse=True)

    return final_scores[:max_n]

def render_email_body_text(data):
    template_loader = FileSystemLoader(searchpath="templates")
    env = Environment(loader=template_loader)

    template = env.get_template("email/to_ship.html")
    rendered_content = template.render(data)
    return rendered_content

async def db_listens_for_new_emails_producer(stoppage_event: asyncio.Event, queue: asyncio.Queue):
    """Deprecated. Until it is decided what approach to take with issue #2, this function will not be used."""

    attempt_interval = 5 # seconds

    # 4. Listen for new emails being added to the database, and add them to the queue
    # Note this only works if MongoDB Node is running in replica set mode, and the database is configured to use Change Streams.
    # This can be done locally by running the following command in the mongo shell:
    # rs.initiate()
    # But only if the initialized node had the proper config that allows for replica sets. Check mongo-setup mongod.conf for commented out example how I did it locally.
    # More info: https://docs.mongodb.com/manual/changeStreams/

    collection = db_client["emails"]
    async with collection.watch() as stream:
        print("Listening for new emails in database.")
        async for change in stream:
            print(type(change))
            retry = True
            while retry:
                if change["operationType"] == "insert":
                    db_object = change["fullDocument"]
                    email = MongoEmail(**db_object)
                    try:
                        queue.put_nowait(email)
                        live_logger.report_to_channel("info", f"Email with id {email.id} added to queue.")
                        break # break out of the while loop
                    except asyncio.QueueFull:
                        live_logger.report_to_channel("warning", f"{queue.qsize()}/{queue.maxsize} emails in messenger queue - waiting for queue to free up space.")
                        await asyncio.sleep(attempt_interval)
                    finally:
                        if stoppage_event.is_set():
                            live_logger.report_to_channel("info", f"Producer closed verified.")
                            return
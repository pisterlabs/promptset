import os
import sqlite3
from sqlite3 import OperationalError

from dotenv import dotenv_values
from fastapi import APIRouter
from openai import OpenAI

from models.travel import Travel

router = APIRouter(tags=["travel"])

config = dotenv_values(".env")

@router.post("/travel")
async def should_travel(travel: Travel) -> None:
    """
        Given a source and destination, return whether the user should travel or not.
        based on if the destination is hotter than the source.

        Args:
            travel (Travel): source and destination
        
        Returns:
            str: whether the user should travel or not.
    """

    source = travel.source
    destination = travel.destination

    # Set the OpenAI API key
    os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]

    # Create the OpenAI chatbot
    client = OpenAI()

    # Define the prompt
    prompt = """
        Write Exactly two SQl queries to get the temperature of the source and destination.

        Query1: SELECT temperature FROM weather WHERE name = {source};
        Query2: SELECT temperature FROM weather WHERE name = {destination};

        return the output in the following format:
        (Query1, Query2)
        """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"{source}, {destination}"}
        ],
        temperature=0
    )

    queries = response.choices[0].message.content.replace("(", "").replace(")", "")
    source_query, destination_query = queries.split(",")

    # Create connection to the SQLite database
    con = sqlite3.connect("data/weather.db")

    # Create cursor
    cur = con.cursor()

    try:
        # Execute the generated queries
        cur.execute(source_query)
        res = cur.fetchone()
        if res is not None:
            source_temp = res
        else:
            source_temp = None

        cur.execute(destination_query)
        res = cur.fetchone()
        if res is not None:
            dest_temp = res[0]
        else:
            dest_temp = None

        con.close() # Close the database connection

    except OperationalError:
        con.close() # Close the database connection
        return "Query Generation Failed!"

    # Check the condition to decide whether the user should travel or not
    if source_temp is None and dest_temp is not None:
        return "Yes You should!"
    if source_temp is not None and dest_temp is None:
        return "No You shouldn't 😟"

    return "District Information is not in the Database"
    
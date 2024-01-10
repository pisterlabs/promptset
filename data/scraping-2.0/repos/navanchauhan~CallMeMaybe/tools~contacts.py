from typing import List
from langchain.agents import tool

from dotenv import load_dotenv
load_dotenv()

import os

CONTACTS = [
    {
        "name": "Greg",
        "phone" : os.getenv("TEST_PHONE_NUMBER"),
        "email": "grsi2038@colorado.edu"
    },
    {
        "name": "Hunter",
        "phone": "+19178737978",
        "email": "hunter.mcrobie@gmail.com"
    }
]

@tool("get_all_contacts")
def get_all_contacts(contact_name: str) -> List[dict]:
    """Returns all contacts in the user's phone book which includes email and phone numbers."""
    return CONTACTS
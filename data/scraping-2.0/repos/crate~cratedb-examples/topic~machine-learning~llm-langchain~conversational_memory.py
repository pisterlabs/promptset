"""
Demonstrate conversational memory with CrateDB.

Synopsis::

    # Install prerequisites.
    pip install -r requirements.txt

    # Start database.
    docker run --rm -it --publish=4200:4200 crate/crate:nightly

    # Run program.
    export CRATEDB_CONNECTION_STRING="crate://crate@localhost/?schema=doc"
    python conversational_memory.py
"""
import os
from pprint import pprint

from langchain.memory.chat_message_histories import CrateDBChatMessageHistory


CONNECTION_STRING = os.environ.get(
    "CRATEDB_CONNECTION_STRING",
    "crate://crate@localhost/?schema=doc"
)


def main():

    chat_message_history = CrateDBChatMessageHistory(
        session_id="test_session",
        connection_string=CONNECTION_STRING,
    )
    chat_message_history.add_user_message("Hello")
    chat_message_history.add_ai_message("Hi")
    pprint(chat_message_history.messages)


if __name__ == "__main__":
    main()

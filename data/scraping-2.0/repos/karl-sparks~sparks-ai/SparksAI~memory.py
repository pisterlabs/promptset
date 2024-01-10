import os

from typing import Optional

from langchain.memory import FileChatMessageHistory

from SparksAI import databases
from SparksAI.models import UserDetails


class AIMemory:
    def __init__(self, database_strategy: databases.DatabaseStrategy) -> None:
        self._convo_mem = {}
        self._user_details = {}

        # initialise db
        self._db = databases.DatabaseContext(strategy=database_strategy)

        self._user_details = {
            user.discord_user_name: user for user in self._db.get_all_rows()
        }

    def get_convo_mem(self, username: str) -> FileChatMessageHistory:
        if username in self._convo_mem:
            return self._convo_mem[username]

        else:
            self._convo_mem[username] = FileChatMessageHistory(f"{username}_memory.txt")

            return self._convo_mem[username]

    def reterive_user_thread_id(self, username: str) -> Optional[str]:
        if username in self._user_details:
            return self._user_details[username].thread_id

        return None

    def update_user_details(self, username: str, thread_id: str) -> None:
        if username not in self._user_details:
            self._user_details[username] = UserDetails(
                discord_user_name=username, thread_id=thread_id
            )
        else:
            self._user_details[username].thread_id = thread_id

        self.sync_users()

    def sync_users(self) -> None:
        for _, user in self._user_details.items():
            self._db.insert_row(user)

        self._user_details = {
            user.discord_user_name: user for user in self._db.get_all_rows()
        }

import os
import sys
import warnings
import random
import pickle
from app.configurations.development.settings import SESSIONS_FILE
from langchain.memory import ChatMessageHistory


class SessionsManagerEngine:
    """
    FIXME: Check how to do it for async in python
    """

    def __init__(self, logger):
        self.sessions = {}
        self.logger = logger

    @property
    def delete_sessions_file(self, path=SESSIONS_FILE):
        """
        Deletes the sessions file
        """
        try:
            os.remove(path)
        except OSError as e:
            # TODO: Add this exception in logging
            # print(f"Error: File doesn't exist - {e}")
            return

    def _create_sessions_folder(self, path):
        """
        Creates the sessions folder
        """
        try:
            os.makedirs(path)
            # TODO: Add this in verbose mode
            # print("The 'sessions' folder has been created.")
        except OSError as e:
            # TODO: Add this exception in logging
            # print(f"Error: Unable to create the 'sessions' folder - {e}")
            return

    def _is_sessions_file_present(self, sessions_file_path=SESSIONS_FILE):
        """
        Checks if the sessions file is present.
        """
        sessions_folder_path = os.path.dirname(SESSIONS_FILE)

        if not os.path.exists(sessions_folder_path):
            self._create_sessions_folder(sessions_folder_path)
            return False
        return os.path.exists(sessions_file_path)

    def load_sessions(self, filename=SESSIONS_FILE):
        """
        Loads the sessions from the session manager
        """
        with open(filename, "rb") as file:
            self.sessions = pickle.load(file)
        # print("***LOADED SESSIONS***: ", self.sessions)

        users_with_sessions = []
        for user, sessions in self.sessions.items():
            user_sessions = {"user": user, "sessions": list(sessions.keys())}
            users_with_sessions.append(user_sessions)

        # TODO only if verbose
        # print("***USERS WITH SESSIONS***: ", users_with_sessions)

    def save_session(self, filename=SESSIONS_FILE):
        """
        Saves the session in the session manager
        FIXME: Save this to a db, write this optimally
        """
        if not self._is_sessions_file_present():
            with open(filename, "wb") as file:
                pickle.dump(self.sessions, file)
        else:
            with open(filename, "rb") as file:
                session_history = pickle.load(file)
                session_history.update(self.sessions)
            with open(filename, "wb") as file:
                pickle.dump(session_history, file)

    def get_session_ids(self, user_id):
        """
        Gets the session ids from the session manager
        """
        if user_id in self.sessions:
            return list(self.sessions[user_id].keys())
        else:
            return []

    def get_session(self, user_id, session_id, username):
        """
        Gets the session from the session manager
        """
        if user_id in self.sessions and session_id in self.sessions[user_id]:
            return self.sessions[user_id][session_id]
        else:
            return self.create_new_session(user_id, session_id, username)

    def update_session(self, user_id, session_id, chat_history, save_session=False):
        """
        Updates the session in the session manager
        """
        if user_id in self.sessions and session_id in self.sessions[user_id]:
            self.sessions[user_id][session_id] = {
                "session_history": chat_history,
            }
        if save_session:
            self.save_session()

    def create_new_session_id(self, user_id, session_id):
        used_session_ids = self.get_session_ids(user_id)

        while True:
            new_session_id = random.randint(1, sys.maxsize)
            if new_session_id not in used_session_ids:
                self.sessions[user_id][new_session_id] = {}
                return new_session_id

    def create_new_session(
        self, user_id, session_id, username, create_session_id=False
    ):
        """
        Creates a new session in the session manager
        """
        if user_id not in self.sessions:
            self.sessions[user_id] = {}
        elif session_id in self.sessions[user_id]:
            self.logger.warning(
                "Session already exists! Returning the existing session."
            )
            # TODO: redundant -> improve it
            self.sessions[user_id]["username"] = username
            return self.sessions[user_id][session_id]["session_history"]
        elif create_session_id or session_id is None:
            session_id = self.create_new_session_id(user_id, session_id)

        self.sessions[user_id][session_id] = {
            "session_history": ChatMessageHistory(),
        }
        self.sessions[user_id]["username"] = username
        return self.sessions[user_id][session_id]["session_history"]

    def get_session_history(self, user_id, session_id, username=None):
        """
        Gets the session history from the session manager
        FIXME: Make it more optimal
        """
        if self._is_sessions_file_present():
            self.load_sessions()

            if user_id in self.sessions and session_id in self.sessions[user_id]:
                return self.sessions[user_id][session_id]["session_history"]

        return self.create_new_session(user_id, session_id, username)

    def clearSessionByUserId(self, user_id, session_id):
        """
        Clears the session for the User in the session manager
        """
        if user_id in self.sessions and session_id in self.sessions[user_id]:
            del self.sessions[user_id][session_id]
            if len(self.sessions[user_id]) == 0:
                del self.sessions[user_id]

    async def clearAllSessions(self):
        self.sessions = {}

    async def showAllSessions(self):
        session_ids = []
        for user_id, user_sessions in self.sessions.items():
            session_ids.extend(list(user_sessions.keys()))
            print("User ID:", user_id)
            print("Session IDs:", list(user_sessions.keys()))
        return session_ids

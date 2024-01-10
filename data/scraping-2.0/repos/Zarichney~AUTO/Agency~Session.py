# /Agency/Session.py

import json
import os
import openai
from Utilities.Config import session_file_name
from Utilities.Log import Debug

class Session:
    def __init__(self, client, prompt, thread_id=None):
        self.prompt: str = prompt
        self.client:openai = client
        self.thread = None
        if thread_id is None:
            self._setup_thread()
        else:
            self._retrieve_thread(thread_id)
            
    def _cancel_runs(self):
        try:
            # Cancel any runs from a previous session
            runs = self.client.beta.threads.runs.list(self.thread.id).data
            for run in runs:
                if (
                    run.status != "completed"
                    and run.status != "cancelled"
                    and run.status != "failed"
                ):
                    self.client.beta.threads.runs.cancel(
                        thread_id=self.thread.id, run_id=run.id
                    )
        except Exception:
            Debug(f"Failed to cancel runs for OpenAI thread_id {self.thread.id}")
            pass
            
    def delete(self):
        if self.thread is None:
            return
        
        self._cancel_runs()
        
        try:
            self.client.beta.threads.delete(self.thread.id)
            Debug("Deleted OpenAI thread")
        except Exception:
            Debug(f"Failed to delete OpenAI thread_id {self.thread.id}")
            pass
            
    def _setup_thread(self):
        if self.thread is not None:
            self._delete_thread()
        self.thread = self.client.beta.threads.create()
        
    def _retrieve_thread(self,thread_id):
        self.thread = self.client.beta.threads.retrieve(thread_id)
        self._cancel_runs()

    def to_dict(self):
        return {
            "prompt": self.prompt,
            "thread_id": self.thread.id
        }

class SessionManager:
    def __init__(self, client, prompt=None, new_session=False):
        self.client = client
        self.sessions: [Session] = []
        self.active_session = None

        # if session_file does not exist or has empty contents, initialize file
        if (
            not os.path.exists(session_file_name)
            or os.stat(session_file_name).st_size == 0
        ):
            self._write_to_session_file()
            
        self._load_from_session_file()

        # Agency established with the user's prompt
        # Perform automatic loading of active_session:
        if prompt is not None:
            
            current_session = self._get_session(prompt)
            
            if new_session and current_session is not None:
                self._remove_session(current_session)
                
            if new_session or current_session is None:
                self.active_session = self._create_session(prompt)
            else:
                self.active_session = current_session
                
    def _remove_session(self, session:Session):
        session.delete()
        # filter out current_session from sessions
        self.sessions = [
            session for session in self.sessions if session != session
        ]
        self._write_to_session_file()
                
    def _load_from_session_file(self):
        self.sessions = []
        with open(session_file_name, "r") as session_file:
            config = json.load(session_file)
            config_sessions = config["sessions"]
            for config_dict in config_sessions:
                self.sessions.append(
                    Session(
                        client=self.client,
                        prompt=config_dict["prompt"],
                        thread_id=config_dict["thread_id"]
                    )
                )

    def _write_to_session_file(self):
        with open(session_file_name, "w") as session_file:
            configurations_dict = [config.to_dict() for config in self.sessions]
            session_file.write(json.dumps({"sessions": configurations_dict}) + "\n")
            
    def _create_session(self, prompt):
        session = Session(self.client, prompt)
        self.sessions.append(session)
        self._write_to_session_file()
        return session

    def _get_session(self, prompt):
        for session in self.sessions:
            if session.prompt == prompt:
                return session
        return None
        
    def get_session(self, prompt):
        self.active_session = self._get_session(prompt)
        if self.active_session is None:
            self.active_session = self._create_session(prompt)
        return self.active_session.thread 
        

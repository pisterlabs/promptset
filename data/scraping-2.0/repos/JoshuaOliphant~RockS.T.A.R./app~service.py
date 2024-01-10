import os
from database_manager import DatabaseManager
from sqlalchemy import create_engine
from openai_api import OpenAI_API


class Service:
    def __init__(self) -> None:
        self._setup_db_manager()
        assistant = self.database_manager.get_assistant()
        self.openai_api = OpenAI_API()
        if assistant is None:
            assistant = self.openai_api.create_assistant()
            self.database_manager.save_assistant(assistant)

    def submit_question(self, question):
        self.openai_api.create_message(question)
        run_created = self.openai_api.create_run()
        self.openai_api.wait_for_run_completion(run_created)
        return self.openai_api.list_messages()

    def _setup_db_manager(self):
        db_name = os.environ.get("DB_NAME", "rockstar")
        db_path = f'{db_name}.sqlite'
        self.engine = create_engine(f'sqlite:///{db_path}')
        self.database_manager = DatabaseManager(self.engine)
        if not os.path.exists(db_path):
            self.database_manager.create_tables()

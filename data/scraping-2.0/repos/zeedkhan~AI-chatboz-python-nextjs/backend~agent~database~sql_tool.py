from backend.gcloud.main import connect_db
import os
from dotenv import load_dotenv
from typing import Any
from langchain.sql_database import SQLDatabase

load_dotenv()
open_ai_key = os.getenv("OPENAI_API_KEY")
print(open_ai_key)


class CustomSQL(SQLDatabase):

    @classmethod
    def from_gcloud(
        cls, connection: str, user: str, password: str, db: str, **kwargs: Any
    ) -> SQLDatabase:
        """Construct a SQLAlchemy engine from URI."""
        conn = connect_db(
            connection=connection,
            user=user,
            password=password,
            db=db
        )

        return cls(conn, **kwargs)

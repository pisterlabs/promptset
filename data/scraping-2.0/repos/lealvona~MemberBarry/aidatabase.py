import datetime
import os
import shortuuid
import sqlite3
import time
import uuid

import chromadb
from chromadb.utils import embedding_functions
# import openai


def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


LONG_TERM_MEMORY_DELAY = 180  # 3 minutes


class AIDatabase:
    def __init__(
            self, session_id=None, db_filename=None, unique_collection=False):
        """Initialize the AIDatabase class."""
        # If no session_id is provided, generate a random one
        if not session_id:
            session_id = str(uuid.uuid4())

        # Assign the session_id to the class
        self.session_id = session_id

        # Open a SQLLite db connection. If a file does not exist, create it.
        if not db_filename:
            db_filename = 'conversations.db'

        self.conn = sqlite3.connect(db_filename)

        self.conn.row_factory = dict_factory

        self.check_tables()

        # Begin VectorDB stuff
        chroma_client = chromadb.PersistentClient(path="./vector_persistance")

        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model_name="text-embedding-ada-002"
        )

        # Allow the use of forever memory, or a unique collection
        if unique_collection:
            collection_name = session_id
        else:
            collection_name = "memberbarry_collection"

        # Get or create the collection with OpenAI embeddings and cosine
        # similarity
        self.collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=openai_ef,
            metadata={"hnsw:space": "cosine"})

    def check_tables(self):
        """Check if the tables exist and create them if they don't."""
        self.create_convo_table()
        self.create_summary_table()

    def create_convo_table(self):
        """Create a table to store every prompt and every response."""
        c = self.conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            system_prompt TEXT,
            user_message TEXT,
            assistant_response TEXT,
            pass INTEGER,
            session_id TEXT
        )""")
        self.conn.commit()

    def create_summary_table(self):
        """Create a table to store summaries."""
        c = self.conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            summary TEXT,
            summary_type TEXT,
            pass INTEGER,
            session_id TEXT
        )""")
        self.conn.commit()

    def get_session_id(self):
        """Return the session_id."""
        return self.session_id

    def get_all_convos(self, session_id=None):
        """Return all conversations for a given session_id by timestamp."""
        if not session_id:
            session_id = self.session_id

        c = self.conn.cursor()
        c.execute(
            """SELECT *
            FROM conversations
            WHERE session_id=?
            ORDER BY timestamp""",
            (session_id,))
        return c.fetchall()

    def get_all_convos_by_pass(self, context_pass, session_id=None):
        """Return all conversations for a given pass and session_id by
        timestamp."""
        if not session_id:
            session_id = self.session_id

        c = self.conn.cursor()
        c.execute(
            """SELECT *
            FROM conversations
            WHERE session_id=?
            AND pass=?
            ORDER BY timestamp""",
            (session_id, context_pass))
        return c.fetchall()

    def get_last_n_convos(self, n, session_id=None):
        """Return the last n conversations for a given session_id by
        timestamp."""
        if not session_id:
            session_id = self.session_id

        c = self.conn.cursor()
        c.execute(
            """SELECT *
            FROM conversations
            WHERE session_id=?
            ORDER BY timestamp DESC
            LIMIT ?""",
            (session_id, n))
        return c.fetchall()

    def get_most_recent_convo(self, session_id=None):
        """Return the most recent conversation for a given session_id."""
        if not session_id:
            session_id = self.session_id

        c = self.conn.cursor()
        c.execute(
            """SELECT *
            FROM conversations
            WHERE session_id=?
            ORDER BY id DESC
            LIMIT 1""",
            (session_id,))
        return c.fetchone()

    def insert_convo(
            self, system_prompt, user_message, assistant_response,
            context_pass, session_id=None):
        """Insert a new conversation."""
        if not session_id:
            session_id = self.session_id

        c = self.conn.cursor()

        timestamp = datetime.datetime.now()

        c.execute(
            """INSERT INTO conversations
            (timestamp, system_prompt, user_message, assistant_response,
             pass, session_id)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (timestamp, system_prompt, user_message, assistant_response,
             context_pass, session_id))
        self.conn.commit()

        c.execute(
            """SELECT last_insert_rowid() as last_id;"""
        )

        return c.fetchone()['last_id']

    def insert_summary(
            self, summary, summary_type, context_pass=None, session_id=None):
        """Insert a new summary."""
        if not session_id:
            session_id = self.session_id

        # If the summary_type is context, or the context pass is missing, set
        # the context_pass to -1 as OoB flag
        if context_pass is None or (summary_type == "context"):
            context_pass = -1

        c = self.conn.cursor()

        timestamp = datetime.datetime.now()

        c.execute(
            """INSERT INTO summaries
            (timestamp, summary, summary_type, pass, session_id)
            VALUES (?, ?, ?, ?, ?)""",
            (timestamp, summary, summary_type, context_pass, session_id))
        self.conn.commit()

        c.execute(
            """SELECT last_insert_rowid() as last_id;"""
        )

        return c.fetchone()['last_id']

    def get_all_summaries_by_type(self, summary_type, session_id=None):
        """Return all summaries for a given session_id of a given summary_type
        by timestamp."""
        if not session_id:
            session_id = self.session_id

        c = self.conn.cursor()
        c.execute(
            """SELECT *
            FROM summaries
            WHERE session_id=? AND summary_type=?
            ORDER BY timestamp""",
            (session_id, summary_type))
        return c.fetchall()

    def get_all_summaries(self, session_id=None):
        """Return all summaries for a given session_id by timestamp."""
        if not session_id:
            session_id = self.session_id

        c = self.conn.cursor()
        c.execute(
            """SELECT *
            FROM summaries
            WHERE session_id=?
            ORDER BY timestamp""",
            (session_id,))
        return c.fetchall()

    def get_last_n_summaries(self, n, session_id=None):
        """Return the last n summaries for a given session_id by timestamp."""
        if not session_id:
            session_id = self.session_id

        c = self.conn.cursor()
        c.execute(
            """SELECT *
            FROM summaries
            WHERE session_id=?
            ORDER BY timestamp DESC
            LIMIT ?""", (session_id, n))
        return c.fetchall()

    def get_most_recent_summary(self, session_id=None):
        """Return the most recent summary for a given session_id."""
        if not session_id:
            session_id = self.session_id

        c = self.conn.cursor()
        c.execute(
            """SELECT *
            FROM summaries
            WHERE session_id=?
            ORDER BY id DESC
            LIMIT 1""",
            (session_id,))
        return c.fetchone()

    def create_context_chain(
            self, session_id=None, context_pass=None, system_prompt=None):
        """Create a chain of prompts and responses from the database."""
        if not session_id:
            session_id = self.session_id

        if context_pass is not None:
            # Get all the convos for the session_id at a given pass
            convos = self.get_all_convos_by_pass(
                context_pass, session_id=session_id)
        else:
            # Get all the convos for the session_id
            convos = self.get_all_convos(session_id=session_id)

        # Check if there are any convos
        if not convos:
            return []

        # Create a chain of prompts and responses, starting with the
        # system_prompt. If no system_prompt is provided, use the first convo
        if not system_prompt:
            convo_chain = [
                {
                    "role": "system",
                    "content": convos[0]['system_prompt']
                }
            ]
        else:  # If a system_prompt is provided, override the existing one
            convo_chain = [
                {
                    "role": "system",
                    "content": system_prompt
                }
            ]

        # Iterate through the convos and add the user messages and assistant
        # responses.
        for convo in convos:
            # Add the user message
            convo_chain.append(
                {
                    "role": "user",
                    "content": convo['user_message']
                }
            )

            # Add the assistant response
            convo_chain.append(
                {
                    "role": "assistant",
                    "content": convo['assistant_response']
                }
            )

        # Return the conversation chain
        return convo_chain

    def get_summary_list(self, session_id=None):
        """Get a list of summaries for a given session_id in chronological '
        order."""
        if not session_id:
            session_id = self.session_id

        c = self.conn.cursor()
        c.execute("""SELECT *
            FROM summaries
            WHERE session_id=?
            ORDER BY timestamp""", (session_id,))

        # Return a list of summary text
        return [summary['summary'] for summary in c.fetchall()]

    # TODO: Is this the most effective method for storing our chats? Should we
    # store the whole message object (i.e. user and assistant messages) as a
    # single item?
    def add_embedding(
            self, user_message, assistant_response, convo_id, session_id=None):
        """Add an embedding to the database."""
        if not session_id:
            session_id = self.session_id

        self.collection.add(
            documents=[user_message, assistant_response],
            metadatas=[
                {
                    "role": "user",
                    'session_id': session_id,
                    'convo_id': convo_id,
                    'timestamp': time.mktime(
                        datetime.datetime.now().timetuple())
                },
                {
                    "role": "assistant",
                    'session_id': session_id,
                    'convo_id': convo_id,
                    'timestamp': time.mktime(
                        datetime.datetime.now().timetuple())
                }
            ],
            ids=[shortuuid.uuid(), shortuuid.uuid()]
        )

    def get_similar_convos(self, query, session_id=None):
        """Get a list of similar responses."""
        # Query for similar past responses, but limit the results to those that
        # were not in the last short period of time.
        where_query = {
            '$and': [
                {"role": "assistant"},
                {'timestamp': {
                    "$lt": time.mktime(
                        datetime.datetime.now().timetuple()) -
                    LONG_TERM_MEMORY_DELAY
                }}
            ]
        }

        # If a session_id is provided, add it to the query
        if session_id:
            where_query['$and'].append({'session_id': session_id})

        # Use the query to get the most similar responses
        results = self.collection.query(
            query_texts=[query],
            n_results=2,
            where=where_query
        )

        messages = []
        for index, distance in enumerate(results['distances'][0]):
            if distance < 0.2:
                where_query = {
                    '$and': [
                        {'convo_id':
                            results['metadatas'][0][index]['convo_id']},
                        {'role': 'user'},
                    ]
                }
                # Get the user message via correlated convo_id
                user_messages = \
                    self.collection.get(where=where_query)['documents']

                if user_messages:
                    # Append the user message and the assistant response to the
                    # messages list
                    messages.append({
                        'role': 'user',
                        'content': user_messages[0]
                    })
                    messages.append({
                        'role': 'assistant',
                        'content': results['documents'][0][index]
                    })

        return messages

    # TODO: This is not properly implemented yet.
    def update_embedding(self, embedding, metadata=None):
        """Update an embedding in the database."""
        pass

    # TODO: This is not properly implemented yet.
    def delete_embedding(self, metadata):
        """Delete an embedding from the database."""
        pass

    @classmethod
    def export_session_rows_to_new_db(cls, session_id=None, db_filename=None):
        """Export all the rows that match a given session_id to a file with a
        unique name based on the session_id."""

        # Set the session_id
        session_id = session_id if session_id else cls.session_id

        # Get all the rows for the session_id
        rows = cls.get_all_rows_for_session_id(cls.session_id)

        # Create a new database file with a unique name based on the
        # session_id.
        db_filename = db_filename if db_filename else str(session_id) + '.db'

        # Create a new database connection
        conn = sqlite3.connect(db_filename)

        # Create a new table
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            system_prompt TEXT,
            user_message TEXT,
            assistant_response TEXT,
            session_id TEXT
        )""")

        # Insert the rows into the new table
        c.executemany(
            'INSERT INTO conversations VALUES (?, ?, ?, ?, ?, ?)', rows)

        # Commit the changes
        conn.commit()

        # Close the connection
        conn.close()

        return db_filename

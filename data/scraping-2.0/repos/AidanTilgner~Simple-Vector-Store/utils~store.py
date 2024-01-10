import sqlite3
from utils.embeddings import OpenAIClient
import sqlite_vss
import array

opc = OpenAIClient()


class Store:
    def __init__(self, db_name):
        self.db_name = db_name
        self.conn = sqlite3.connect(db_name)
        self.conn.enable_load_extension(True)
        self.cursor = self.conn.cursor()
        sqlite_vss.load(self.conn)

    def get_name(self):
        return self.db_name

    def reset_db(self):
        self.cursor.execute("DROP TABLE IF EXISTS knowledge_base")
        self.cursor.execute("DROP TABLE IF EXISTS vss_knowledge_base")
        self.create_knowledge_base_table()
        self.create_vss_table()

    def create_knowledge_base_table(self):
        """
        Creates the knowledge base table.
        """
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge_base (
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                type TEXT NOT NULL DEFAULT 'txt'
            )
            """
        )

    def insert_into_knowledge_base(self, path, title, content, filetype):
        """
        Insert a new item into the knowledge base.
        """
        self.cursor.execute(
            """
            INSERT INTO knowledge_base (path, title, content, type)
            VALUES (?, ?, ?, ?)
            """,
            (path, title, content, filetype),
        )

        title_embedding_binary = array.array(
            "f", opc.generate_embedding(title)
        ).tobytes()
        content_embedding_binary = array.array(
            "f", opc.generate_embedding(content)
        ).tobytes()

        self.cursor.execute(
            """
            INSERT INTO vss_knowledge_base (rowid, title_embedding, content_embedding)
            VALUES (last_insert_rowid(), ?, ?)
            """,
            (title_embedding_binary, content_embedding_binary),
        )
        self.conn.commit()

    def create_vss_table(self):
        """
        Creates the vector search table.
        """
        self.cursor.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS vss_knowledge_base USING vss0(
                title_embedding(1536),
                content_embedding(1536)
            );
            """
        )

    def search_similar_items(self, query, search_in="content"):
        """
        Search for items similar to the given query.

        :param query: The query string to search for.
        :param search_in: The column to search in ('title' or 'content').
        :return: A list of tuples containing the rowid and similarity distance of the matching items.
        """
        # Generate the embedding for the query
        query_embedding = array.array("f", opc.generate_embedding(query)).tobytes()

        # Choose the column to search against
        column = "title_embedding" if search_in == "title" else "content_embedding"

        # Execute the vector search query
        self.cursor.execute(
            f"""
            SELECT rowid, distance
            FROM vss_knowledge_base
            WHERE vss_search({column}, ?)
            ORDER BY distance ASC
            LIMIT 10;
            """,
            (query_embedding,),
        )

        # Fetch and return the results
        return self.cursor.fetchall()

    def search_and_map_similar_items(self, query: str, search_in="content", limit=10):
        """
        Search for items similar to the given query, and map the results to the corresponding rows in the knowledge base.

        :param query: The query string to search for.
        :param search_in: The column to search in ('title' or 'content').
        :return: A list of tuples containing the rowid and similarity distance of the matching items.
        """
        # Generate the embedding for the query
        query_embedding = array.array("f", opc.generate_embedding(query)).tobytes()

        # Choose the column to search against
        column = "title_embedding" if search_in == "title" else "content_embedding"

        # Step 1: Execute the vector search query to get rowids
        self.cursor.execute(
            f"""
            SELECT rowid, distance
            FROM vss_knowledge_base
            WHERE vss_search({column}, ?)
            ORDER BY distance ASC
            LIMIT ?;
            """,
            (query_embedding, limit),
        )
        search_results = self.cursor.fetchall()

        results = []
        if search_results:
            self.cursor.execute(
                f"""
                    WITH SearchResults(rowid, distance) AS (
                        VALUES {','.join(f'({row[0]}, {row[1]})' for row in search_results)}
                    )
                    SELECT knowledge_base.rowid, title, content, SearchResults.distance
                    FROM knowledge_base
                    INNER JOIN SearchResults ON knowledge_base.rowid = SearchResults.rowid
                    ORDER BY SearchResults.distance ASC;
                    """
            )
            results = self.cursor.fetchall()

        return results

    def get_all_titles(self):
        """
        Gets all of the titles in the knowledge base.
        """
        self.cursor.execute(
            """
            SELECT title FROM knowledge_base
            """
        )
        return self.cursor.fetchall()

    def get_all(self):
        """
        Get all items in the knowledge base.
        """
        self.cursor.execute(
            """
            SELECT id, title, path, content, type FROM knowledge_base
            """
        )
        return self.cursor.fetchall()

    def get_by_id(self, identifier):
        """
        Get the item with the given id.
        """
        self.cursor.execute(
            """
            SELECT id, title, path, content, type FROM knowledge_base
            WHERE id = ?
            """,
            (identifier,),
        )
        return self.cursor.fetchone()

    def update_item(self, identifier: int, title: str, content: str):
        """
        Update the item with the given id, title, and content.
        This function will delete the old entry in the vector search table and create a new one,
        as well as updating the knowledge base.
        """
        # Update the knowledge base
        self.cursor.execute(
            """
            UPDATE knowledge_base
            SET title = ?, content = ?
            WHERE id = ?
            """,
            (title, content, identifier),
        )

        # Generate embeddings for the new title and content
        title_embedding_binary = array.array(
            "f", opc.generate_embedding(title)
        ).tobytes()
        content_embedding_binary = array.array(
            "f", opc.generate_embedding(content)
        ).tobytes()

        # Delete the old entry in the VSS table
        self.cursor.execute(
            """
            DELETE FROM vss_knowledge_base
            WHERE rowid = ?
            """,
            (identifier,),
        )

        # Insert a new entry in the VSS table
        self.cursor.execute(
            """
            INSERT INTO vss_knowledge_base (title_embedding, content_embedding, rowid)
            VALUES (?, ?, ?)
            """,
            (title_embedding_binary, content_embedding_binary, identifier),
        )

        # Commit the transaction
        self.conn.commit()

    def delete_item(self, identifier: int):
        """
        Delete the item with the given id.
        """
        self.cursor.execute(
            """
            DELETE FROM knowledge_base
            WHERE id = ?
            """,
            (identifier,),
        )
        self.cursor.execute(
            """
            DELETE FROM vss_knowledge_base
            WHERE rowid = ?
            """,
            (identifier,),
        )
        self.conn.commit()

    def get_id_from_title(self, title):
        """
        Get the id of the item with the given title.
        """
        self.cursor.execute(
            """
            SELECT id FROM knowledge_base
            WHERE title = ?
            """,
            (title,),
        )
        return self.cursor.fetchone()[0]

    def get_id_from_path(self, path):
        """
        Get the id of the item with the given path.
        """
        self.cursor.execute(
            """
            SELECT id FROM knowledge_base
            WHERE path = ?
            """,
            (path,),
        )
        return self.cursor.fetchone()[0]

    def get_entry_from_path(self, path):
        """
        Get the entry with the given path.
        """
        self.cursor.execute(
            """
            SELECT id, title, path, content, type FROM knowledge_base
            WHERE path = ?
            """,
            (path,),
        )
        return self.cursor.fetchone()

    @staticmethod
    def get_content_summary(content: str, length: int) -> str:
        """Returns a summary of the content."""
        return content[:length] + ("..." if len(content) > length else "")

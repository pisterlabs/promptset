import openai
import os
import singlestoredb as s2
from typing import Optional
from world import World

class Database():
    """Interface for the database which stores the context used by the AI.
    The context includes:
    - rules;
    - objects;
    - characters.
    """

    async def fill(self, world: World):
        """Fills the database with data from the given world"""
        raise NotImplementedError()

    async def query(self, task: str, error: Optional[str] = None) -> list[str]:
        """Queries context for the given task, optionally with the error message of the previous action if it failed"""
        raise NotImplementedError()

class DumbDatabase(Database):
    """Dumb database which dumps all of the context to the AI"""

    def __init__(self):
        self.world = None

    async def fill(self, world: World):
        self.world = world

    async def query(self, task: str, error: Optional[str] = None) -> list[str]:
        assert self.world is not None, "Database must be filled before querying"

        context = []
        context += list(map(lambda x: x.rule(), self.world.interactions.values()))
        context += list(map(lambda id, x: f"There is a '{x.type}' named '{id}'.", self.world.objects.keys(), self.world.objects.values()))
        #context += list(map(lambda id: f"There is a character named '{id}'.", self.world.characters.keys()))

        print()
        print(f"Context queried from task '{task}' and error {error}")
        for ctx in context:
            print(f" - {ctx}")
        print()

        return context

class SingleStoreDatabase(Database):
    """Database which uses SingleStore as a backend"""

    def __init__(self,
                 encoding: str,
                 model: str,
                 host: str,
                 port: int,
                 user: str,
                 password: str,
                 database: str):
        self._encoding = encoding
        self._model = model
        self._conn = s2.connect(host=host, port=port, user=user, password=password, database=database)

        self.filled = False
        self._id = -1

    def new_id(self):
        self._id += 1
        return self._id

    async def get_embedding(self, text):
        """Returns the vector for semantic search, using the OpenAI embedding API"""
        return (await openai.Embedding.acreate(input=[text], model=self._model))["data"][0]["embedding"] # type: ignore

    async def get_embeddings(self, vector):
        """get_embedding but mapped to a vector of inputs"""
        result = []
        for text in vector:
            result.append(await self.get_embedding(text))
        return result

    async def fill(self, world: World):
        """Fills the database with data from the given world"""

        print("Filling database...")

        context = []
        context += list(map(lambda x: x.rule(), world.interactions.values()))
        context += list(map(lambda id, x: f"There is a '{x.type}' named '{id}'.", world.objects.keys(), world.objects.values()))
        #context += list(map(lambda id: f"There is a character named '{id}'.", world.characters.keys()))

        with self._conn.cursor() as cursor:
            cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS info(
                        id INT not null PRIMARY KEY,
                        context TEXT,
                        vector blob
                        );
                    """
            )
            cursor.execute("DELETE FROM info;")

            query = """INSERT INTO info VALUES """
            for ctx, vector in zip(context, await self.get_embeddings(context)):
                query += f"""({self.new_id()}, "{ctx}", JSON_ARRAY_PACK('{vector}')),"""
            query = query[:-1] + ";"
            cursor.execute(query)

        self.filled = True
        print("Database filled")

    async def query(self, task: str, error: Optional[str] = None) -> list[str]:
        """Queries context for the given task, optionally with the error message of the previous action if it failed"""
        assert self.filled, "Database must be filled before querying"

        goal_vector = await self.get_embedding(task + ("" if error is None else " " + error))

        context_filtered = []

        with self._conn.cursor() as cursor:
            cursor.execute(
                f"""
                    SELECT id, context, dot_product(vector, JSON_ARRAY_PACK('{goal_vector}')) AS score
                    FROM info
                    ORDER BY score DESC
                    LIMIT 10;
                """
            )
            results = cursor.fetchall()

        for row in results:
            _, filtered, _ = row # type: ignore
            context_filtered += [
                filtered,
            ]

        print()
        print(f"Context queried from task '{task}' and error {error}")
        for ctx in context_filtered:
            print(f" - {ctx}")
        print()

        return context_filtered

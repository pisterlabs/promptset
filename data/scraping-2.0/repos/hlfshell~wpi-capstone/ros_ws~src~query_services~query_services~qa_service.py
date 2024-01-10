from capstone_interfaces.srv import LLM, PlannerQuery
from query_services import tools, database_functions
import os
import openai
from langchain.embeddings import OpenAIEmbeddings
import pandas as pd

import sqlite3

import rclpy
import builtin_interfaces
from math import sqrt
from rclpy.node import Node

from capstone_interfaces.msg import StateObject, Item

from typing import List, Tuple

openai.api_key = os.getenv("OPENAI_API_KEY")


class QuestionAnswerService(Node):
    def __init__(self):
        super().__init__("question_answer_service")
        self.conn = database_functions.create_connection(self, r"state_db.db")
        self.srv = self.create_service(PlannerQuery, "general_query", self.query)

        # Listen for pickup messages to remove from database
        self.__pickup_subscription = self.create_subscription(
            Item, "/litterbug/pickup", self.__pickup_callback, 10
        )

        print("inited")

    def query(
        self, question_request: PlannerQuery.Request, response: PlannerQuery.Response
    ):
        """
        From queried question, returns response location and x,y,z
        """
        question = question_request.question

        cur = self.conn.cursor()
        cur.execute("SELECT * FROM objects")
        rows = cur.fetchall()

        input_format = ["id", "description", "location", "x", "y", "z", "time seen"]

        tools.save_input_to_json(rows, input_format)
        tools.embed_documents_in_json_file("state.json")
        df = pd.read_json("state.json", orient="index")

        res = tools.search_embeddings(df, question, n=25, pprint=True, n_lines=1)

        quantile = res["similarities"].quantile(0.75)
        res = res[res["similarities"] > quantile]
        print("prior", res)

        state_objects: List[StateObject] = []
        for index, row in res.iterrows():
            # self.get_logger().info(f">>> {index}, {row}")
            state_objects.append(
                StateObject(
                    id=row["id"],
                    description=row["description"],
                    location=row["location"],
                    x=row["x"],
                    y=row["y"],
                    z=row["z"],
                )
            )
        response.objects = state_objects

        return response

    def __pickup_callback(self, msg: Item):
        """
        Removes item from database when picked up
        """
        print("pickup callback")
        cur = self.conn.cursor()

        # First we need to get the id of the item by
        # searching for its label (description in the
        # db) and then comparing the location; if
        # it's close enough we consider it a match
        cur.execute(
            """
                    SELECT
                        id,
                        x,
                        y
                    FROM objects
                    WHERE description = ?
                    """,
            (msg.label,),
        )
        rows = cur.fetchall()

        # If we don't find any matches, we can't
        # delete anything
        if len(rows) == 0:
            return

        # Otherwise, we need to check if any of the
        # matches are close enough to the item
        # we're trying to delete
        for row in rows:
            if self.__distance((row[1], row[2]), (msg.x, msg.y)) < 0.10:
                cur.execute(
                    """
                    DELETE FROM objects
                    WHERE id = ?
                    """,
                    (row[0],),
                )
                self.conn.commit()
                return

    def __distance(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        """
        __distance calculates the distance between two points
        """
        return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def main():
    rclpy.init()

    qa = QuestionAnswerService()
    rclpy.spin(qa)

    rclpy.shutdown()


if __name__ == "__main__":
    main()

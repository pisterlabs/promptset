from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from openai_tools import get_embedding, get_importance_of_interaction, get_insights
from config import get_logger
import numpy as np

logger = get_logger(__name__)

class Memory:
    def __init__(self, db, name, partition=None):
        self.db = db
        self.name = name
        self.partition = partition

    def upload_message_response_pair(self, message, response):
        importance = get_importance_of_interaction(message, response)
        embedding = get_embedding(message + response)
        metadata = {
            "message": message,
            "response": response,
            "importance": importance,
            "timestamp": datetime.now(),
        }
        self.db.insert_memory(name=self.name, embedding=embedding, metadata=metadata, partition=self.partition)

    def insert_insight(self, insight):
        embedding = get_embedding(insight["content"])
        metadata = {
            "insight": insight["content"],
            "importance": insight["importance"],
            "timestamp": datetime.now(),
        }
        self.db.insert_memory(name=self.name, embedding=embedding, metadata=metadata, partition=self.partition)

    def reflect(self, messages):
        insights = get_insights(messages)
        futures = []
        with ThreadPoolExecutor() as executor:
            for insight in insights:
                futures.append(executor.submit(self.insert_insight, insight))
        for future in futures:
            future.result()

    def search(self, vector, n=100):
        message_response_pairs = self.db.recall_memory(
            name=self.name, vector=vector, n=n, partition=self.partition
        )
        results = [
            {
                "message": result["metadata"]["message"]
                if "message" in result["metadata"]
                else None,
                "response": result["metadata"]["response"]
                if "response" in result["metadata"]
                else None,
                "insight": result["metadata"]["insight"]
                if "insight" in result["metadata"]
                else None,
                "timestamp": result["metadata"]["timestamp"],
                "importance": result["metadata"]["importance"],
                "similarity": result["score"],
            }
            for result in message_response_pairs
        ]

        if results:
            # Compute the values for each dimension
            days_since_values = []
            importance_values = []
            similarity_values = []
            for result in results:
                timestamp = datetime.strptime(
                    result["timestamp"], "%Y-%m-%d %H:%M:%S.%f"
                )
                days_since = (datetime.now() - timestamp).days
                days_since_values.append(np.exp(-0.99 * days_since))
                importance_values.append(result["importance"])
                similarity_values.append(result["similarity"])

            # Calculate the min and max values for each dimension
            min_days_since, max_days_since = min(days_since_values), max(
                days_since_values
            )
            min_similarity, max_similarity = min(similarity_values), max(
                similarity_values
            )
            # Calculate the min and max values for importance separately for insights and non-insights
            min_importance_insight, max_importance_insight = min(
                x["importance"] for x in results if x["insight"]
            ), max(x["importance"] for x in results if x["insight"])

            min_importance_no_insight, max_importance_no_insight = min(
                x["importance"] for x in results if not x["insight"]
            ), max(x["importance"] for x in results if not x["insight"])

            # Apply min-max scaling and compute the scaled score
            epsilon = 1e-8
            for i, result in enumerate(results):
                days_since_scaled = (days_since_values[i] - min_days_since) / (
                    max_days_since - min_days_since + epsilon
                )
                similarity_scaled = (similarity_values[i] - min_similarity) / (
                    max_similarity - min_similarity + epsilon
                )

                # Scale importance based on whether it was an insight or not
                if result["insight"]:
                    importance_scaled = (
                        importance_values[i] - min_importance_insight
                    ) / (max_importance_insight - min_importance_insight + epsilon)
                else:
                    importance_scaled = (
                        importance_values[i] - min_importance_no_insight
                    ) / (
                        max_importance_no_insight - min_importance_no_insight + epsilon
                    )

                result["score"] = (
                    1 / 3 * days_since_scaled
                    + 1 / 3 * importance_scaled
                    + 1 / 3 * similarity_scaled
                )

            # Sort the results based on the score in descending order
            results.sort(key=lambda x: x["score"], reverse=True)

        # Return the top n results
        return results[:n]

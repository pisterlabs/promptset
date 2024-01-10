# https://python.langchain.com/docs/integrations/vectorstores/redis

import os
import dotenv
# from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores.redis import Redis
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

dotenv.load_dotenv()

REDIS_URL = os.environ["REDIS_URL"]
INDEX_NAME = "problem"
# embeddings = CohereEmbeddings(model="embed-english-light-v3.0")
# embeddings = OllamaEmbeddings(model="llama2:7b")
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")


def initialize_database():
    """
    Initializes the Redis database with three pieces of data and creates the schema file
    :return: Database object
    """
    texts = [
        "The construction industry is indubitably one of the significant contributors to global waste, contributing approximately 1.3 billion tons of waste annually, exerting significant pressure on our landfills and natural resources. Traditional construction methods entail single-use designs that require frequent demolitions, leading to resource depletion and wastage. Herein, we propose an innovative approach to mitigate this problem: Modular Construction. This method embraces recycling and reuse, taking a significant stride towards a circular economy.   Modular construction involves utilizing engineered components in a manufacturing facility that are later assembled on-site. These components are designed for easy disassembling, enabling them to be reused in diverse projects, thus significantly reducing waste and conserving resources.  Not only does this method decrease construction waste by up to 90%, but it also decreases construction time by 30-50%, optimizing both environmental and financial efficiency. This reduction in time corresponds to substantial financial savings for businesses. Moreover, the modular approach allows greater flexibility, adapting to changing needs over time.  We believe, by adopting modular construction, the industry can transit from a 'take, make and dispose' model to a more sustainable 'reduce, reuse, and recycle' model, driving the industry towards a more circular and sustainable future. The feasibility of this concept is already being proven in markets around the globe, indicating its potential for scalability and real-world application.",
        "I'm sure you, like me, are feeling the heat - literally! With World Health Organization declaring climate change as ""the greatest threat to global health in the 21st century"", we're in a race against time to move away from fossil fuels to more efficient, less polluting electrical power. But as we take bold leaps into a green future with electric cars and heating, we're confronted with a new puzzle - generating enough electrical power without using fossil fuels! Imagine standing on a green hill, not a single towering, noisy windmill in sight, and yet, you're surrounded by wind power generation! Using existing, yet under-utilized technology, I propose a revolutionary approach to harness wind energy on a commercial scale, without those ""monstrously large and environmentally damaging windmills"". With my idea, we could start construction tomorrow and give our electrical grid the jolt it needs, creating a future where clean, quiet and efficient energy isn't a dream, but a reality we live in. This is not about every home being a power station, but about businesses driving a green revolution from the ground up!",
        "The massive shift in student learning towards digital platforms has resulted in an increased carbon footprint due to energy consumption from data centers and e-waste from obsolete devices. Simultaneously, physical books are often produced, used once, and then discarded, leading to waste and deforestation. Implement a ""Book Swap"" program within educational institutions and local communities. This platform allows students to trade books they no longer need with others who require them, reducing the need for new book production and hence, lowering the rate of resource depletion. Furthermore, the platform could have a digital component to track book exchanges, giving users credits for each trade, which they can accrue and redeem. This system encourages and amplifies the benefits of reusing and sharing resources, thus contributing to the circular economy.   By integrating gamification, getting students and parents involved and providing an easy-to-use platform, the program could influence a cultural shift towards greater resource value appreciation and waste reduction. In terms of the financial aspect, less reliance on purchasing new books could save money for students, parents and schools."]
    metadata = [
        {
            "novelty_score": 4,
            "utility_score": 2,
            "surprise_score": 4,
        },
        {
            "novelty_score": 5,
            "utility_score": 5,
            "surprise_score": 3,
        },
        {
            "novelty_score": 3,
            "utility_score": 3,
            "surprise_score": 3,
        }
    ]

    rds = Redis.from_texts(
        texts,
        embeddings,
        metadatas=metadata,
        redis_url=REDIS_URL,
        index_name=INDEX_NAME,
    )
    rds.write_schema("redis_schema.yaml")
    return rds


def existing_database():
    """
    Connects to a previously initialized Redis vector database using schema file
    :return: Database object
    """
    rds = Redis.from_existing_index(
        embeddings,
        index_name=INDEX_NAME,
        redis_url=REDIS_URL,
        schema="redis_schema.yaml",
    )
    return rds


def query_database(rds, query: str, k: int):
    """
    Performs a similarity search on the query
    :param rds: Database object
    :param query: Query text
    :param k: Top k results
    :return: List of results
    """
    results = rds.similarity_search(query, k=k)
    return results


def add_vectors(rds, document: list, metadata: list):
    """
    Adds vectors to Redis database
    :param rds: Database object
    :param document: Query used, must be list ["example"]
    :param metadata: Metadata (scores) from the document, must be list [{"test": "example"}]
    :return: None
    """
    rds.add_texts(document, metadata)
    return None


def determine_score(rds, query: str):
    """
    Determines query score based on top k results
    :param rds: Database object
    :param query: Query text
    :return: Dictionary containing scores
    """
    k = 3
    novelty_score = 0
    utility_score = 0
    surprise_score = 0
    results = query_database(rds, query, k)
    # print(results)
    for i in range(0, k):
        metadata = results[i].metadata
        novelty_score += float(metadata["novelty_score"])
        utility_score += float(metadata["utility_score"])
        surprise_score += float(metadata["surprise_score"])
    return_dict = {
        "novelty_score": round(novelty_score / k, 4),
        "utility_score": round(utility_score / k, 4),
        "surprise_score": round(surprise_score / k, 4)
    }
    return return_dict


if __name__ == "__main__":
    initialize_database()
    # rds = existing_database()
    # query = "In congested cities like Berlin, one of the significant challenges faced by carsharing users is the difficulty in finding a parking spot and handing over the vehicle, resulting in extended driving distances, increased costs, frustration, and augmented fuel consumption."
    # print(determine_score(rds, query))

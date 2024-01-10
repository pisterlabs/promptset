"""Utilities to manipulate reviews"""
import random
import uuid, datetime
from langchain.vectorstores import Cassandra, VectorStore

from common_constants import (
    FEATURED_VOTE_THRESHOLD,
    REVIEWS_TABLE_NAME,
    REVIEW_VECTOR_TABLE_NAME,
)
from utils.models import HotelReview, UserProfile

from utils.ai import get_embeddings
from utils.db import get_session, get_keyspace

from typing import List

# LangChain VectorStore abstraction to interact with the vector database
review_vectorstore = None


def get_review_vectorstore(embeddings, is_setup: bool = False):
    global review_vectorstore
    if review_vectorstore is None:
        review_vectorstore = Cassandra(
            embedding=embeddings,
            table_name=REVIEW_VECTOR_TABLE_NAME,
            partition_id="will-always-be-overridden",
            partitioned=True,
            skip_provisioning=not is_setup,
        )
    return review_vectorstore


# ### SELECTING REVIEWS

# Prepared statements for selecting reviews
select_recent_reviews_stmt = None
select_featured_reviews_stmt = None
select_review_count_by_hotel_stmt = None


# Entry point to select reviews for the general (base) hotel summary
def select_general_hotel_reviews(hotel_id: str) -> List[HotelReview]:
    session = get_session()
    keyspace = get_keyspace()

    review_dict = {}

    global select_recent_reviews_stmt
    if select_recent_reviews_stmt is None:
        select_recent_reviews_stmt = session.prepare(
            f"SELECT id, title, body, rating FROM {keyspace}.{REVIEWS_TABLE_NAME} WHERE hotel_id = ? LIMIT 3"
        )

    rows_recent = session.execute(select_recent_reviews_stmt, (hotel_id,))
    for row in rows_recent:
        review_dict[row.id] = HotelReview(
            id=row.id, title=row.title, body=row.body, rating=row.rating
        )

    global select_featured_reviews_stmt
    if select_featured_reviews_stmt is None:
        select_featured_reviews_stmt = session.prepare(
            f"SELECT id, title, body, rating FROM {keyspace}.{REVIEWS_TABLE_NAME} "
            f" WHERE hotel_id = ? and featured = 1 LIMIT 3"
        )

    rows_featured = session.execute(select_recent_reviews_stmt, (hotel_id,))
    for row in rows_featured:
        review_dict[row.id] = HotelReview(
            id=row.id, title=row.title, body=row.body, rating=row.rating
        )

    return list(review_dict.values())


def select_hotel_reviews_for_user(
    hotel_id: str, user_travel_profile_summary: str
) -> List[HotelReview]:
    review_store = get_review_vectorstore(
        embeddings=get_embeddings(),
    )

    review_data = review_store.similarity_search_with_score_id(
        query=user_travel_profile_summary, k=3, partition_id=hotel_id
    )

    reviews = [
        HotelReview(
            title=review_doc.metadata["title"].strip(),
            body=extract_review_body_from_doc_text(
                review_doc.page_content, review_doc.metadata["title"]
            ),
            # was: body=review_doc.page_content[len(review_doc.metadata["title"].strip())+1:].strip(),
            rating=float(review_doc.metadata["rating"]),
            id=review_id,
        )
        for review_doc, _, review_id in review_data
    ]

    return reviews


def select_review_count_by_hotel(hotel_id: str) -> int:
    session = get_session()
    keyspace = get_keyspace()

    global select_review_count_by_hotel_stmt
    if select_review_count_by_hotel_stmt is None:
        select_review_count_by_hotel_stmt = session.prepare(f"SELECT COUNT(id) AS num_reviews FROM {keyspace}.{REVIEWS_TABLE_NAME} "
                                                            f"WHERE hotel_id = ?")

    row = session.execute(select_review_count_by_hotel_stmt, (hotel_id,)).one()
    return row.num_reviews


# Extracts the review body from the text found in the document,
# cutting out the title and colon and removing any leading / trailing spaces from title and body
def extract_review_body_from_doc_text(review_doc_text: str, review_title: str) -> str:
    end_title_index = len(review_title.strip()) + 1
    return review_doc_text[end_title_index:].strip()


# ### ADDING REVIEWS

# Prepared statement for inserting reviews
insert_review_stmt = None


# Entry point for when we want to add a review
# - Generates an id for the new review
# - Stores the review in the non-vectorised table
# - Embeds the review and then stores it in the vectorised table
def insert_review_for_hotel(
    hotel_id: str, review_title: str, review_body: str, review_rating: int
):
    review_id = generate_review_id()
    insert_into_reviews_table(
        hotel_id, review_id, review_title, review_body, review_rating
    )
    insert_into_review_vector_table(
        hotel_id, review_id, review_title, review_body, review_rating
    )


def generate_review_id():
    return uuid.uuid4().hex


def format_review_content_for_embedding(title: str, body: str) -> str:
    return f"{title}: {body}"


def choose_featured(num_upvotes: int) -> int:
    if num_upvotes > FEATURED_VOTE_THRESHOLD:
        return 1
    else:
        return 0


# Inserts a new review into the non-vectorized reviews table
def insert_into_reviews_table(
    hotel_id: str,
    review_id: str,
    review_title: str,
    review_body: str,
    review_rating: int,
):
    session = get_session()
    keyspace = get_keyspace()

    global insert_review_stmt
    if insert_review_stmt is None:
        insert_review_stmt = session.prepare(
            f"""INSERT INTO {keyspace}.{REVIEWS_TABLE_NAME} (hotel_id, date_added, id, title, body, rating, featured) 
                    VALUES (?, ?, ?, ?, ?, ?, ?)"""
        )

    date_added = datetime.datetime.now()
    featured = choose_featured(random.randint(1, 21))

    session.execute(
        insert_review_stmt,
        (
            hotel_id,
            date_added,
            review_id,
            review_title,
            review_body,
            review_rating,
            featured,
        ),
    )


# Inserts a new review into the vectorized reviews table, using a VectorStore of type Cassandra from LangChain
def insert_into_review_vector_table(
    hotel_id: str,
    review_id: str,
    review_title: str,
    review_body: str,
    review_rating: int,
):
    review_store = get_review_vectorstore(
        embeddings=get_embeddings(),
    )

    review_metadata = {
        "hotel_id": hotel_id,
        "rating": review_rating,
        "title": review_title,
    }

    review_store.add_texts(
        texts=[format_review_content_for_embedding(review_title, review_body)],
        metadatas=[review_metadata],
        ids=[review_id],
        partition_id=hotel_id,
    )

import pandas as pd
import openai
import re


# Define the filter_predictions function
def filter_predictions(
    user_id, predictions_df, books_df, tags_df, book_tags_df, user_request
):
    # Define the system message to establish the assistant's role
    system_message = "You are a helpful assistant. Provide a response in a simple and structured format suitable for processing by a program.\
        Only return a list of book genres, themes, authors, and tags that are relevant to a user's request, as comma-separated values.\
            No need to return the response like so: Genres: Fantasy\nThemes: Formula One, Racing, Motorsport\nAuthors: N/A\nTags: Sports Fantasy, Racing, Magical Realism, Car Racing, Motorsport Fantasy\
                Bur rather like so: Fantasy, Formula One, Racing, Motorsport, Sports Fantasy, Racing, Magical Realism, Car Racing, Motorsport Fantasy.\
                    Do not return N/A or Not available, simply return nothing"

    # Define the user message with the actual prompt
    user_message = user_request

    # Call GPT API
    book_response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
    )

    # Return the relevant attributes
    relevant_attributes = book_response["choices"][0]["message"]["content"].split(",")

    # Replace newlines and multiple subsequent spaces (if any) with a single comma, then split
    formatted_response = (
        book_response["choices"][0]["message"]["content"]
        .replace("\n", ",")
        .replace("  ", " ")
    )

    # Split by comma and strip whitespace
    all_attributes = [
        item.strip() for item in formatted_response.split(",") if item.strip() != ""
    ]

    # Filter out category names like 'Genres:', 'Themes:', 'Authors:', and 'Tags:'
    category_headers = [
        "Genres:",
        "book genres:",
        "Genres:",
        "Themes:",
        "themes:",
        "Authors:",
        "authors:",
        "Tags:",
        "tags:",
    ]
    relevant_attributes = [
        attr for attr in all_attributes if attr not in category_headers
    ]

    # Filter tags that match the relevant attributes
    relevant_tags = tags_df[
        tags_df["tag_name"].apply(
            lambda x: any(
                re.search(attr.strip(), x, re.IGNORECASE)
                for attr in relevant_attributes
            )
        )
    ]
    relevant_tag_ids = relevant_tags["tag_id"].unique()

    # Filter book_tags that match the relevant tag IDs
    relevant_book_tags = book_tags_df[book_tags_df["tag_id"].isin(relevant_tag_ids)]
    relevant_book_ids = relevant_book_tags["goodreads_book_id"].unique()

    # Filter books that match the relevant book IDs and attributes
    filtered_books = books_df[
        books_df["goodreads_book_id"].isin(relevant_book_ids)
        | books_df["authors"].apply(
            lambda x: any(
                re.search(attr.strip(), x, re.IGNORECASE)
                for attr in relevant_attributes
            )
        )
        | books_df["title"].apply(
            lambda x: any(
                re.search(attr.strip(), x, re.IGNORECASE)
                for attr in relevant_attributes
            )
        )
    ]

    # Filter predictions for the user that match the relevant book IDs
    filtered_predictions = predictions_df[
        (predictions_df["user_id"] == user_id)
        & (predictions_df["book_id"].isin(filtered_books["book_id"]))
    ]

    return filtered_predictions


# Define the rerank_predictions function
def rerank_predictions(filtered_predictions, books_df, user_request):
    # After filtering predictions, merge with books data to include book details
    filtered_books = books_df[
        books_df["goodreads_book_id"].isin(filtered_predictions["book_id"])
    ]
    filtered_predictions_with_details = filtered_predictions.merge(
        filtered_books[["goodreads_book_id", "title", "authors"]],
        left_on="book_id",
        right_on="goodreads_book_id",
    )

    # Reranking prompt sent to GPT API
    reranking_prompt = f"""
    Re-rank these books based on the user's preference, only return the top 10.
    output fomrat should follow this example:
    "1. Book ID: 5038, Title: The Pillars of Creation (Sword of Truth, #7), Authors: Terry Goodkind, Prediction: 0.8330672979354858 (Fantasy)
    2. Book ID: 9567, Title: Half Asleep in Frog Pajamas, Authors: Tom Robbins, Prediction: 0.8262556195259094 (Romance)
    3. Book ID: 5368, Title: Forever Amber, Authors: Kathleen Winsor, Prediction: 0.7773409485816956 (Romance)
    Reasoning: Note that I prioritized books that are known for romance or have strong romantic elements, and disregarded
    those that focus on other genres like fantasy or thrillers, unless they are known to blend romance into the narrative
    substantially. If the user is strictly looking for pure romance novels, some books such as "Harry Potter and the
    Prisoner of Azkaban" and non-romance focused thrillers have been left out of the top 10.
    "
    Filtered Recommendations with Details:
    """

    for index, row in filtered_predictions_with_details.iterrows():
        reranking_prompt += f"Book ID: {row['book_id']}, Title: {row['title']}, Authors: {row['authors']}, Prediction: {row['prediction']}\n"

    reranking_prompt += f"User Preference: {user_request}"

    # Sort by prediction score in descending order and take the top 10
    filtered_predictions_with_details = filtered_predictions_with_details.sort_values(
        by="prediction", ascending=False
    ).head(10)

    # Call the chat completion API for reranking
    reranking_response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": reranking_prompt},
        ],
    )

    # Extract re-ranked recommendations from the response
    reranked_recommendations = reranking_response["choices"][0]["message"][
        "content"
    ].strip()

    return reranked_recommendations

""" Build the vector index from the markdown files in the directory. """
import json
import logging
import os
import sys
import traceback

import openai
import pandas as pd
from chat_db_helper import Chat, ChatVectorDB
from dotenv import load_dotenv
from tqdm import tqdm

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Load the .env file
load_dotenv()

# Set up the OpenAI API key
assert os.getenv("OPENAI_API_KEY"), "Please set your OPENAI_API_KEY environment variable."
openai.api_key = os.getenv("OPENAI_API_KEY")


def main() -> None:
    """Build the vector index from the markdown files in the directory."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Each conversation grouped into a single thread_id
    chats_df = pd.read_csv(os.path.join(base_dir, ".content", "chats", "chats.csv"))
    chats_df.head()
    chats_df = chats_df.fillna("")
    print("Number of all conversations: ", len(chats_df))

    # Cache of embeddings from data
    # The embedding for each conversation with its thread_id (Note: not all embeddings were generated for the chat text)
    # Create a temp index of the chats
    chats_index = {}
    for _, row in tqdm(chats_df.iterrows(), desc="Creating temporary chats index"):
        chats_index[row["thread_id"]] = row["chat_text"]

    embeddings_df = pd.read_csv(os.path.join(base_dir, ".content", "chats", "chats-embeddings-ada-002.csv"))
    embeddings_df.head()
    embeddings_cache = {}
    for index, row in tqdm(embeddings_df.iterrows(), desc="Creating a cache of embeddings"):
        chat_text = chats_index[row["thread_id"]]
        embedding = json.loads(row["embedding"])
        embeddings_cache[chat_text] = embedding

    chats = []
    if "summary" not in chats_df.columns:
        chats_df["summary"] = [""] * len(chats_df)

    # NOTE: THIS WILL USE SIGNIFICANT OPENAI API CREDITS TO SUMMARIZE ALL THE CONVERSATIONS
    for index, row in tqdm(chats_df.iterrows(), total=len(chats_df)):
        try:
            if row.get("summary"):
                chat = Chat(
                    thread_id=row["thread_id"],
                    text=row["chat_text"],
                    summary=row["summary"],
                    embeddings_cache=embeddings_cache,
                )
            else:
                chat = Chat(thread_id=row["thread_id"], text=row["chat_text"], embeddings_cache=embeddings_cache)
            chats_df.at[index, "summary"] = chat.summary
            chats.append(chat)
            # Save the summary to the csv, overwrite the file
            if index % 250 == 0:
                chats_df.to_csv(os.path.join(base_dir, ".content", "chats", "chats.csv"), index=False, quotechar='"')
        except KeyboardInterrupt as keyi:
            print("Keyboard Interrupt")
            raise keyi
        except Exception:  # pylint: disable=bare-except
            traceback.print_exc()

    # Save remaining ones (%10)
    chats_df.to_csv(os.path.join(base_dir, ".content", "chats", "chats.csv"), index=False, quotechar='"')

    # Bulk insert them
    chat_vector_db = ChatVectorDB()
    chat_vector_db.add(chats)


if __name__ == "__main__":
    main()

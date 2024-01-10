import torch
from transformers import T5Tokenizer, T5EncoderModel
import tiktoken
import numpy as np
from sklearn.cluster import KMeans
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()
import os
import pandas as pd

from create_embedding_tsne import getEmbeddingFromText, reduce_dimensions


client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

# Initialize the T5 model and tokenizer
encoder_model = T5EncoderModel.from_pretrained("t5-large").to("cuda")
tokenizer = T5Tokenizer.from_pretrained("t5-large")

conv_csv_cols = [
    "conversation_id",
    "model",
    "conversation",
    "turn",
    "language",
    "openai_moderation",
    "redacted",
]
msg_csv_cols = [
    "message_id",
    "model",
    "turn",
    "role",
    "content",
    "toxicity",
    "openai_moderation",
    "vector",
]


def conversation_to_messages(
    conv: list[dict], id: str, model: str, openai_moderation: str
) -> pd.DataFrame:
    """
    Convert a conversation represented as a list of dictionaries of messages into a DataFrame of messages.

    Args:
        conv (list[dict]): List of dictionaries representing a conversation.
        id (str): Unique identifier for the conversation.
        model (str): LLM name associated with the conversation.
        openai_moderation (str): Moderation/toxicity information for all messages in conversation.

    Returns:
        pd.DataFrame: DataFrame containing messages extracted from the conversation.
    """
    df = pd.DataFrame(
        columns=msg_csv_cols  # embedding and openai_moderation can be added as columns
    )
    messages = []
    for i in range(len(conv)):
        message_turn = i // 2 + 1
        is_toxic = openai_moderation[i]["flagged"]
        embedding = getEmbeddingFromText(conv[i]["content"])
        new_message = {
            "message_id": id + "_" + str(i),
            "model": model,
            "turn": message_turn,
            "role": conv[i]["role"],
            "content": conv[i]["content"],
            "toxicity": is_toxic,
            "openai_moderation": openai_moderation[i],
            "vector":  embedding # if conv[i]["role"]=="assistant" else None,
            # conditional moderation value can be added for message of toxic conversations or None in other cases
        }
        messages.append(new_message)
    df = pd.concat([df, pd.DataFrame(messages, columns=msg_csv_cols)])
    df.set_index(["message_id"]).index.is_unique
    return df


def create_message_csv(model: str, save_path: str, load_path: str) -> None:
    """
    Process original LLM-specific conversation data and create a CSV file containing individual extracted messages.

    Args:
        model (str): LLM name associated with the conversation data.
        save_path (str): The directory where the processed dataset will be stored
        load_path (str): The directory where the original/unprocessed dataset is stored.
    """
    # Loads the original dataset containing conversations
    df_orig = pd.read_csv(os.path.join(load_path, f"{model}.csv"))
    df_proc = pd.DataFrame(
        columns=msg_csv_cols,
    )
    for i in range(len(df_orig)):
        conv_list = eval(df_orig.conversation[i].replace("}", "},"))
        moderation = eval(
            (df_orig.openai_moderation[i]).replace("}", "},").replace("},,", "},")
        )

        df_proc = pd.concat(
            [
                df_proc,
                conversation_to_messages(
                    conv=conv_list,
                    id=df_orig.conversation_id[i],
                    model=df_orig.model[i],
                    openai_moderation=moderation,
                ),
            ],
            ignore_index=True,
        )

    # Dimensionality reduction of the embeddings stored in 'vector' column
    embeddings = df_proc['vector'].tolist()
    embeddings_array = np.array(embeddings)
    reduced_embeddings = reduce_dimensions(embeddings=embeddings_array, n_components=3)
    df_proc['vector'] = reduced_embeddings.tolist()

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=10, random_state=0, n_init="auto").fit(reduced_embeddings)
    
    # Assigning cluster labels to the DataFrame
    cluster_labels = kmeans.labels_
    df_proc['cluster'] = np.nan  # Initialize the column with NaN values
    df_proc.loc[df_proc['vector'].notnull(), 'cluster'] = cluster_labels  # Assign clusters only to rows with embeddings

    # Retrieve original message text for the cluster centers
    for cluster_id, center in enumerate(kmeans.cluster_centers_):
        closest_idx = np.argmin(np.linalg.norm(reduced_embeddings - center, axis=1))
        closest_message = df_proc.iloc[closest_idx]['content']
        cluster_summary = createTopicSummary(closest_message)
        print(f"Cluster {cluster_id} summary: {cluster_summary}")
        # Add the summary as a new column, labeled as 'cluster_summary'
        df_proc.loc[df_proc['cluster'] == cluster_id, 'cluster_summary'] = cluster_summary


    # Saving the CSV with cluster information
    df_proc.to_csv(
        os.path.join(save_path, f"{model}.csv"),
        index=False,
    )
    print(model, ":", len(df_proc))



def createTopicSummary(message_text):
    prompt = f"Summarize the following message in less than 7 words:\n\n{message_text}"
    response = client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.8,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
    return response.choices[0].message.content

def num_tokens_from_string(string: str, encoding_name: str="gpt2") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# llm_models = [
#     "palm-2",
#     "gpt-3.5-turbo",
#     "gpt4all-13b-snoozy",
# ]
# for llm in llm_models:
#     create_message_csv(model=llm)

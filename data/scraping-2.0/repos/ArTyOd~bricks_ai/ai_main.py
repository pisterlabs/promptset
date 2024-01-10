import subprocess
import pandas as pd
import numpy as np
import json
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity
import openai
import pinecone
import datetime
import json
import streamlit as st
from google.oauth2 import service_account
from google.cloud import storage
import tiktoken


pinecone_api_key = st.secrets["PINECONE_API_KEY_Bricks"]
pinecone_environment = st.secrets["PINECONE_environment_Bricks"]
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Set the OpenAI API key
openai.api_key = openai_api_key

# Create API client for Google Cloud Storage
credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
client = storage.Client(credentials=credentials)

# Bucket name and file path
bucket_name = "bucket_g_cloud_service_1"

messages = [
    {
        "role": "system",
        "content": "Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"",
    }
]
selected_categories = []
session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def save_chat_history(chat_history, filename):
    file_path = f"bricks/chat_history/{filename}"
    bucket = client.bucket(bucket_name)
    chat_log_blob = bucket.blob(file_path)
    chat_log_content = json.dumps(chat_history)
    chat_log_blob.upload_from_string(chat_log_content, content_type="application/json")


def clear_chat_history():
    global messages, session_id
    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    messages = [
        {
            "role": "system",
            "content": "Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"",
        }
    ]


def save_chat_log(chat_log, filename):
    folder_path = "log/"
    with open(folder_path + filename, "w") as f:
        json.dump(chat_log, f)


# Define the function to count tokens
def count_tokens(text):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))


def load_index():
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
    index_name = "bricks"
    if not index_name in pinecone.list_indexes():
        raise KeyError(f"Index '{index_name}' does not exist.")
    return pinecone.Index(index_name)


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not presently implemented for model {model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )


def create_context(question, selected_categories, index, max_len=1500, size="ada"):
    """
    Create a context for a question by finding the most similar context from the Pinecone index
    """

    # Get the embeddings for the question
    q_embed = openai.Embedding.create(input=question, engine="text-embedding-ada-002")["data"][0]["embedding"]

    # Query Pinecone index
    res = index.query(q_embed, filter={"category": {"$in": selected_categories}}, top_k=5, include_metadata=True)

    context_details = []
    returns = []
    cur_len = 0

    # Iterate through results, sorted by score (ascending), and add the text to the context until the context is too long
    for match in sorted(res["matches"], key=lambda x: x["score"]):
        # Get the length of the text
        text_len = match["metadata"]["n_tokens"]

        # Add the length of the text to the current length
        cur_len += text_len + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else, add it to the text that is being returned
        returns.append(match["metadata"]["content"])

        # Add the context information to the context_details list
        context_details.append(
            {
                "score": match.get("score", None),
                "category": match["metadata"].get("category", None),
                "content": match["metadata"].get("content", None),
                "topic": match["metadata"].get("topic", None),
                "url": match["metadata"].get("url", None),
                "token": match["metadata"].get("n_tokens", None),
            }
        )

    print(f"How much Contexts found: {len(returns)} \n----------------\n ")
    # Return the context and context_details
    return "\n\n###\n\n".join(returns), context_details


def fallback_reframe_question(messages, original_question, model="gpt-3.5-turbo", temperature=0.3, max_tokens=100):
    """
    Rephrase the original question to be more specific, given the context of the chat history (messages).
    """

    # Create a copy of the original messages
    # messages_copy = messages.copy()
    messages_reframe = []
    # print(f"messages before the prompting reframing: {messages_reframe} \n----------------\n ")
    # Add the rephrase instruction message to the messages_reframe

    rephrase_instruction = f"Your task is to rephrase the user's question below, considering the given CHAT HISTORY. Ensure the rephrased question is clear, concise, and works as a standalone query. Do not provide any additional information, summary, or explanation. If you cannot rephrase the question, simply respond with 'I don't know'. /n CHAT HISTORY: {[message.items() for message in messages]}"
    messages_reframe.append({"role": "system", "content": f"{rephrase_instruction}"})
    messages_reframe.append({"role": "user", "content": original_question})
    print(f"messages after the prompting reframing: {messages_reframe} \n----------------\n ")
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages_reframe,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        rephrased_question = response["choices"][0]["message"]["content"].strip()
        if "don't know" in rephrased_question:
            return original_question
        return rephrased_question
    except Exception as e:
        print(e)
        print("An error occurred while rephrasing the question.")
        return ""


def engineer_prompt(question, selected_categories, index, max_len, model, reframing=False):
    """
    Answer a question based on the most similar context from the Pinecone index
    """
    global messages
    if "mail" in selected_categories:
        selected_categories.remove("mail")
        context, context_details = create_context(question, selected_categories, index, max_len)
        context_mail, context_details_mail = create_context(question, ["mail"], index, max_len=1000)
        # For strings, just use the + operator
        context_combined = context + " " + context_mail  # Added a space in between for separation

        # If context_details are dictionaries, you can merge them.
        # This will overwrite any duplicate keys in context_details with values from context_details_mail
        context_details_combined = context_details + context_details_mail
    else:
        context, context_details = create_context(question, selected_categories, index, max_len)
        context_combined = context
        context_details_combined = context_details

    prompt = [{"role": "assistant", "content": f"Context: {context_combined}"}, {"role": "user", "content": f"{question}"}]
    return prompt, context_details_combined


def answer_question(
    model="gpt-3.5-turbo",
    instruction="Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"",
    index="",
    categories=[],
    question="what?",
    reframing=True,
    size="ada",
    max_tokens=150,
    max_len=1500,
    temperature=0.5,
    debug=False,
    stop=None,
    callback=None,
):
    global messages

    selected_categories = categories
    messages[0] = {"role": "system", "content": f"{instruction}"}
    prompt, context_details = engineer_prompt(
        question=question, selected_categories=selected_categories, index=index, max_len=max_len, reframing=reframing, model=model
    )
    messages += prompt

    print(f"messages before the prompting OpenAO: {messages} \n----------------\n ")
    # If debug, print the raw model response
    if debug:
        print(f"question = {question}")
        # print("Context:\n" + context)
        print(messages)
        print("\n\n")

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        output_text = ""
        for chunk in response:
            if "role" in chunk["choices"][0]["delta"]:
                continue
            elif "content" in chunk["choices"][0]["delta"]:
                r_text = chunk["choices"][0]["delta"]["content"]
                output_text += r_text
                if callback:
                    callback(r_text)

        if callback:
            system_message = {"role": "assistant", "content": output_text}

            prompt_tokens = num_tokens_from_messages(messages)
            completion_tokens = num_tokens_from_messages([system_message])
            total_tokens = prompt_tokens + completion_tokens
            messages = [d for d in messages if not (d.get("role") == "assistant" and "Context" in d.get("content"))]
            messages.append(system_message)
            print(f"messages after the prompt : {messages}  \n----------------\n")
            chat_log_filename = f"{session_id}_chat_log.json"
            save_chat_history(messages, chat_log_filename)
            print(f"{total_tokens =} {prompt_tokens =} { completion_tokens =}")
            return messages, context_details, prompt_tokens, completion_tokens, total_tokens

        return response["choices"][0]["message"]["content"], context_details
    except Exception as e:
        print(e)
        print(f"something is wrong")
        return "", "", 0, 0, 0


# An Ai which helps to develop a long-term memory solution for large language models using external knowledge storage, vector embeddings, and sub-modules. The system should be independent of the language model and easily interchangeable. The focus is on creating an efficient and optimized memory structure that prevents slowdowns and increased expenses over time.

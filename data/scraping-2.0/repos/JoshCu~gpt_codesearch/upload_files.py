import os
import openai
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Supabase credentials
url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# OpenAI credentials
openai.api_key = os.getenv("OPENAI_API_KEY")


def upload_file_info(folder_path: str):
    # Retrieve all existing file paths from the database in a single query
    existing_files_response = (
        supabase.table("code_vectors").select("file_path").execute()
    )
    existing_file_paths = {
        record["file_path"] for record in existing_files_response.data
    }

    # Prepare data for new files
    new_files_data = []
    for root, dirs, files in os.walk(folder_path):
        if ".git" in root or "node_module" in root:
            continue

        for file in files:
            file_path = os.path.join(root, file)
            if file_path not in existing_file_paths:
                new_files_data.append(
                    {
                        "file_path": file_path,
                        "file_name": os.path.basename(file_path),
                        "explanation": None,
                        "explanation_embedding": None,
                    }
                )

    # Batch insert new files
    if new_files_data:
        response = supabase.table("code_vectors").insert(new_files_data).execute()


def update_explanations_and_embeddings():
    # Retrieve records with empty embeddings
    data = (
        supabase.table("code_vectors")
        .select("*")
        .filter("explanation", "is", "null")
        .execute()
    )

    for record in data.data:
        file_path = record["file_path"]

        # Read the content of the file
        try:
            # Attempt to read the file with UTF-8 encoding
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
        except UnicodeDecodeError:
            try:
                # If UTF-8 fails, try with a more inclusive encoding like 'latin-1'
                with open(file_path, "r", encoding="latin-1") as file:
                    content = file.read()
            except Exception as e:
                print(f"Failed to read file {file_path}: {e}")
                continue

        # Step Two: Explain the code
        explanation = explain_code(content, file_path)

        # Step Three: Embed the explanation
        embedding = embed_explanation(explanation)

        # Update the record
        updated_data = {"explanation": explanation, "explanation_embedding": embedding}
        response = (
            supabase.table("code_vectors")
            .update(updated_data)
            .eq("id", record["id"])
            .execute()
        )


def explain_code(content, file_path):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant who explains code and summarises documentation.",
        },
        {
            "role": "user",
            "content": f"What does this code do or documentation describe? for context it's path is {file_path}\n The contents is:\n{content}\n",
        },
    ]
    if len(content) > 40000:
        return "This file is too large to explain."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106", messages=messages, max_tokens=300
        )
        # Extracting the last assistant's message
        assistant_message = response["choices"][0]["message"]["content"]
        return assistant_message.strip()
    except Exception as e:
        print(f"Error in generating explanation for {file_path}: {e}")
        return ""


def embed_explanation(explanation):
    try:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=truncate_string_tokens(explanation, 8191),
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error in embedding explanation: {e}")
        return []


def truncate_string_tokens(text, max_length):
    # Truncate the string to a certain number of tokens
    return text[:max_length]


if __name__ == "__main__":
    # First upload the file information
    upload_file_info("./repos")

    # Then update the records with explanations and embeddings
    update_explanations_and_embeddings()

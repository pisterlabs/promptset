import psycopg2
import json
import openai
import time
import re
from google.cloud import secretmanager

client = secretmanager.SecretManagerServiceClient()
name = f"projects/vidbox-7d2c1/secrets/openai_secret/versions/latest"
response = client.access_secret_version(request={"name": name})
openai.api_key = response.payload.data.decode("UTF-8")

def get_long_plot():
    ids = []
    with open('long_plots.txt', 'r') as file:
        for line in file:
            #print(f"Processing line: {line.strip()}")
            match = re.search(r'id: (\d+)', line)
            if match:
                #print(f"Extracted ID: {match.group(1)}")
                ids.append(match.group(1))
    return ids


def build_embeddings():
    conn = psycopg2.connect(
        database="vidbox-backend_development",
        host="localhost",
        user="vishaalganesan",
        password="vish",
        port="5433")

    # Create a cursor to execute SQL commands.
    cursor = conn.cursor()

    id_tuple = tuple(get_long_plot())

    # Execute the SQL command to select all rows from the 'movie_infos_top_rated' table.
    cursor.execute("SELECT * FROM movie_infos_top_rated")

    # Fetch the results.
    results = cursor.fetchall()

    # Fetch the column names.
    columns = [desc[0] for desc in cursor.description]

    # Create a dictionary that maps movie IDs to their embedding vectors.
    movie_embeddings = {}
    requests_count = 0
    token_count = 0
    start_time = time.time()

    print("Processing started...")

    for idx, row in enumerate(results):
        text = ""
        for column_name, value in zip(columns, row):
            text += str(column_name) + ": " + str(value) + " | "

        if 4*len(text) >= 8191:
            text = ""
            for column_name, value in zip(columns, row):
                if column_name == 'plot': continue
                text += str(column_name) + ": " + str(value) + " | "
        try:
            movie_embedding = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=text # Use the concatenated text as input
            )

            movie_embeddings[row[columns.index("id")]] = movie_embedding['data']

            # Increment request counter and update token count
            requests_count += 1
            token_count += movie_embedding['usage']['total_tokens']

            # Print progress
            if idx % 100 == 0:
                print(f"Processed {idx} rows. Current token count: {token_count}")

            # Check for limits: 2800 requests or 1,000,000 tokens
            if requests_count == 2900 or token_count >= 999900:
                print(f"Rate limits reached. Requests: {requests_count}, Tokens: {token_count}. Waiting...")
                elapsed_time = time.time() - start_time
                sleep_time = max(60 - elapsed_time, 0)
                time.sleep(sleep_time)
                requests_count = 0
                token_count = 0
                start_time = time.time()
        except Exception as e:
            with open('long_plots.txt', 'a') as file:  # Use 'a' for append mode
                file.write(f"{text}\n")
            continue
    # Close the cursor and connection.
    cursor.close()
    conn.close()
    # Save the embedding vectors to a file.
    with open("embeddings2.json", "w") as f:
        json.dump(movie_embeddings, f)

    print("Processing complete.")
    print(f"Total keys: {len(movie_embeddings.keys())}")

#print(len(get_long_plot()))
#build_embeddings()

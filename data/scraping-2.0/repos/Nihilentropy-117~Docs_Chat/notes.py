import json
import textwrap
import time
import psycopg2
from openai import OpenAI
from telebot import types
import keys
from keys import postgres_connection_params

client = OpenAI(api_key=keys.openai_key)



def speak(message):
    assistant_statement = "You are an AI that writes the script for a personal assistant. You are always helpful and accurate, and never too verbose."

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": assistant_statement},
            {"role": "user", "content": f"How would the personal assistant uniquely phrase the following sentence? [{message}] Reply with only that sentence."}
        ]

    )
    return completion.choices[0].message.content

def open_db():
    return psycopg2.connect(**postgres_connection_params)

def get_tables():
    db = open_db()  # Fixed: Added parentheses to call the function
    cursor = db.cursor()
    # Query to get the list of tables
    query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
    cursor.execute(query)
    table_names = cursor.fetchall()
    cursor.close()
    db.close()
    return [table[0] for table in table_names]
def get_vector_search(query_vector, table="documents", limit=10):
    db = open_db()
    with db as conn:
        with conn.cursor() as cur:
            # Extract the actual numeric vector from the query_vector dictionary
            embedding_vector = query_vector['data'][0]['embedding']  # This should be a list of floats

            # Determine the dimension of your vectors
            vector_dim = len(embedding_vector)

            # Format your query to include the distance
            query = f"""
                SELECT text, author, source, part, json_data, vector_embedding <-> CAST(%s AS vector({vector_dim})) AS distance
                FROM {table}
                ORDER BY distance
                LIMIT %s;
            """

            # Execute the query with the embedding_vector and limit
            cur.execute(query, (embedding_vector, limit))
            results = []
            for row in cur.fetchall():
                text, author, source, part, json_data, distance = row
                if json_data:
                    json_data = json.loads(json_data)  # Parse JSON string to dictionary
                results.append({"content": text, "author": author, "source": source, "part": part, "json_data": json_data, "distance": distance})
            return results


def create_embeddings_ember(text):
    import requests

    # Define the API endpoint and headers
    url = 'https://api.llmrails.com/v1/embeddings'
    headers = {
        'X-API-KEY': f'{keys.llm_rail_api}',  # Replace {token} with your actual API key
        'Content-Type': 'application/json'
    }

    # Define the payload
    data = {
        "input": [f"{text}"],
        "model": "embedding-english-v1"  # equals to ember-v1
    }

    # Make the POST request
    response = requests.post(url, headers=headers, json=data)

    # Print the response
    return (response.json())


def format_text_to_width(text, width):
    return textwrap.fill(text, width)


def inquire(query, table="books", limit=4):
    vector_q = create_embeddings_ember(query)

    results = get_vector_search(vector_q, table=table, limit=limit)

    references = ""
    for result in results:
        title = result['source']
        author = result['author']
        part = result['part']
        content = result['content']
        references += f"{title}: Part:{part}\n{author}\n{content} \n\n\n"


    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. "
                                          "Read the sources, then answer the question, quoting and referencing the sources extensively. "
                                          "Always answer the question to the best of your ability, backing it up with the sources. "
                                          "If it seems to you the sources cannot help directly, discuss how they can be applied to the further answering of the question."
                                          "It is important that links are made between all of the sources and the answer."
                                          "The user cannot see the sources that you can, so quote directly anything you reference."},
            {"role": "user", "content": f"References: \n {references}"},
            {"role": "user", "content": f"Query: {query}"}
        ]
    )

    response = (completion.choices[0].message.content)

    return [references, response, results]



def flow(message, bot):
    markup = types.ReplyKeyboardMarkup(one_time_keyboard=True)
    tables = get_tables()

    for table in tables:
        markup.add(types.KeyboardButton(table))

    msg = bot.reply_to(message, speak('What table would you like me to search?'), reply_markup=markup)
    bot.register_next_step_handler(msg, get_user_query, bot=bot)


def get_user_query(message, bot):
    msg = bot.reply_to(message, speak('What do you want to know?'))
    source = message.text
    bot.register_next_step_handler(msg, search, source=source, bot=bot)


def search(message, bot, source):
    query = message.text
    references, response, results = inquire(query, source, limit=2)
    i = 1
    for result in results:
        title = result['source']
        author = result['author']
        part = result['part']
        content = result['content']
        bot.send_message(message.chat.id, f"Source {i}:\n{title}: Part:{part}\n{author}\n{content}")
        i += 1
        time.sleep(2)
    time.sleep(2)
    bot.send_message(message.chat.id, f"Analysis:\n\n{response}")

if __name__ == "__main__":
    import start_bot
    start_bot.main()

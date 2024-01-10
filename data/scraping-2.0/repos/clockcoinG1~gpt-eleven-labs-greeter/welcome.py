import sys
from elevenlabs import generate, stream, set_api_key, VoiceSettings
import os
import openai
import datetime
import sqlite3


settings = VoiceSettings(
    stability=0.21, similarity_boost=0.1, style=0.1, use_speaker_boost=True, speaking_rate=0.5, pitch=0.24, volume=0.99
)
openai.api_key = os.getenv("OPENAI_API_KEY")
set_api_key("")

# Create a new database or connect to existing one
conn = sqlite3.connect('chat_history.db')
c = conn.cursor()

# Create table to store chat history
c.execute('''CREATE TABLE IF NOT EXISTS chat_history
			 (id INTEGER PRIMARY KEY, role TEXT, content TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')


def create_and_seed_db(db_name='chat_history.db'):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()

    # Create table to store chat history
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history
				 (id INTEGER PRIMARY KEY, role TEXT, content TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    # Seed the database with initial data if necessary
    c.execute("SELECT COUNT(*) FROM chat_history")
    if c.fetchone()[0] == 0:
        # Add seed data
        seed_data = [
            ('user', 'Hello, how are you?'),
            ('assistant', 'I am fine, thank you! How can I assist you today?'),
            # Add more seed messages if needed
        ]
        c.executemany("INSERT INTO chat_history (role, content) VALUES (?, ?)", seed_data)
        conn.commit()

    # Close the connection to the database
    conn.close()


create_and_seed_db()


def save_message_to_db(role, content, conn):
    c = conn.cursor()
    c.execute("INSERT INTO chat_history (role, content) VALUES (?, ?)", (role, content))
    conn.commit()


def get_previous_messages(conn):
    c = conn.cursor()
    c.execute("SELECT role, content FROM chat_history ORDER BY timestamp DESC LIMIT 15")
    return [{'role': role, 'content': content} for role, content in c.fetchall()]


def gettext(conn):
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    previous_messages = get_previous_messages(conn)

    completion = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        temperature=2,
        top_p=0.25,
        max_tokens=4096,
        messages=previous_messages
        + [
            {
                "role": "system",
                "content": (
                    "You are to generate NYC facts about the engineering, buildings, and crime history of the city and"
                    " randomly select 15 to tell Anton who has just arrived home from being out, don't be afraid to be"
                    " funny."
                ),
            },
        ],
    )

    save_message_to_db("system", completion.choices[0].message['content'], conn)
    content = ""
    for chunk in openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        temperature=2,
        top_p=0.25,
        max_tokens=4096,
        stream=True,
        messages=previous_messages
        + [
            {
                "role": "system",
                "content": (
                    "SYSTEM: Greet Anton home and compile  information to enlighten Anton's day from the content: \n"
                    f" {completion.choices[0].message['content']}\n\n[Speech for Greeting Anton at: {date} ] :"
                ),
            },
        ],
    ):
        try:
            if chunk['choices'][0]['delta']['content']:
                content += chunk['choices'][0]['delta']['content']
                print(chunk['choices'][0]['delta']['content'], end="")
                yield chunk['choices'][0]['delta']['content']
        except:
            save_message_to_db("assistant", content, conn)
            pass


# Use the conn object in the gettext function
audio_stream = generate(
    text=gettext(conn),
    voice="hjUljYvGJsI6RPf2Vzc6",
    model="eleven_monolingual_v1",
    stream=True,
    latency=3,
    stream_chunk_size=2024,
)

stream(audio_stream)

# Close the connection to the database when done
conn.close()

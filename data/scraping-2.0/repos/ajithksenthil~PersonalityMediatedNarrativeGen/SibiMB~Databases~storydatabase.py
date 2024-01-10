import sqlite3
from sqlite3 import Error
import openai

openai.api_key = "your_openai_api_key"


# Create a database connection to a SQLite database
def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(f'successful connection with {db_file}')
    except Error as e:
        print(e)

    return conn

# Create a table from the create_table_sql statement
def create_table(conn):
    create_story_table = """CREATE TABLE IF NOT EXISTS stories (
                                id INTEGER PRIMARY KEY,
                                user_id INTEGER NOT NULL,
                                story_title TEXT NOT NULL,
                                content TEXT NOT NULL,
                                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                            );"""

    try:
        cursor = conn.cursor()
        cursor.execute(create_story_table)
    except Error as e:
        print(e)

# Create a new story
def add_story(conn, story):
    sql = '''INSERT INTO stories(user_id, story_title, content)
             VALUES(?, ?, ?)'''
    cur = conn.cursor()
    cur.execute(sql, story)
    conn.commit()
    return cur.lastrowid

# Update story by appending new content
def update_story(conn, story_id, new_content):
    sql = '''UPDATE stories
             SET content = content || ?,
                 timestamp = CURRENT_TIMESTAMP
             WHERE id = ?'''
    cur = conn.cursor()
    cur.execute(sql, (new_content, story_id))
    conn.commit()

# Get a story by id without length limit
def get_story_by_id(conn, story_id):
    cur = conn.cursor()
    cur.execute("SELECT * FROM stories WHERE id=?", (story_id,))

    return cur.fetchone()

# get story by id with the length limit
def get_story_by_id(conn, story_id, length_limit=None):
    cur = conn.cursor()
    cur.execute("SELECT * FROM stories WHERE id=?", (story_id,))

    story = cur.fetchone()

    if length_limit and len(story[3]) > length_limit:
        summarized_content = summarize_text(story[3], length_limit)
        story = (story[0], story[1], story[2], summarized_content, story[4])

    return story


# Get all stories
def get_all_stories(conn):
    cur = conn.cursor()
    cur.execute("SELECT * FROM stories")

    return cur.fetchall()


def main():
    database = "stories.db"

    conn = create_connection(database)
    if conn is not None:
        create_table(conn)
    else:
        print("Error! Cannot create the database connection.")

    # Example usage:
    story_id = add_story(conn, (1, "Story Title", "Story content."))
    update_story(conn, story_id, " Updated story content.")
    story = get_story_by_id(conn, story_id)
    print(story)

    all_stories = get_all_stories(conn)
    print(all_stories)


def summarize_text(text, length_limit):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"Please provide a summary of the following text in {length_limit} characters or fewer: {text}",
        max_tokens=length_limit,
        n=1,
        stop=None,
        temperature=0.5,
    )

    summary = response.choices[0].text.strip()
    return summary



if __name__ == '__main__':
    main()



# story_id = 1
# length_limit = 100
# summarized_story = get_story_by_id(conn, story_id, length_limit)
# print(summarized_story)

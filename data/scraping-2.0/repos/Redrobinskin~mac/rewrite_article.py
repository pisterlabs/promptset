import openai
import sys
import sqlite3

# Read the article from standard input
article = sys.stdin.read()

# Split the article into lines
lines = article.split('\n')

# Ensure that there are at least two lines (for the title and content)
if len(lines) < 2:
    print("Error: The input must contain at least two lines (a title and content).")
    sys.exit(1)

# Get the original title and content# Get the original title and content
original_title = lines[1].strip().replace("Title: ", "")
original_content = " ".join(lines[3:]).replace("Content: ", "")

# Ensure that the title and content are not blank
if not original_title or not original_content:
    print("Error: The title and content cannot be blank.")
    sys.exit(1)

# Print out the title and content to debug
print("Title:", original_title)
print("Content:", original_content)

# Set up the OpenAI API with your API key
openai.api_key = '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'

# Use the chat models, which work well for multi-turn conversations
response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo-16k",
  messages=[
        {"role": "system", "content": "You are a professional, witty cybersecurity consultant, rewrite the users article AND TITLE in a captivating article of no more than 2500 words. Utilize components of markdown syntax to make the article more visually appealing, such as headings, block quotes, code snippets, strong, italic, and table of contents where necessary. Big focus on USING block quotes, hints & tips formatting in markdown. Also can use tables, mermaid diagrams. The rewritten articles should focus on being both an enjoyable read and helping the reader be safer by understanding what to do or not to do. Finish with a 'lessons learnt' or 'what you can do to protect yourself' section where appropriate. Output in markdown format."},
        {"role": "user", "content": original_content}
    ]
)

# Get the rewritten article
rewritten_article = response['choices'][0]['message']['content']

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('articles.db')

# Create a cursor object
cursor = conn.cursor()

# Create table if it doesn't exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS rewritten_articles (
        id INTEGER PRIMARY KEY,
        original_title TEXT,
        original_content TEXT,
        rewritten_content TEXT
    )
''')

# Insert rewritten data into table
cursor.execute('''
    INSERT INTO rewritten_articles (original_title, original_content, rewritten_content) VALUES (?, ?, ?)
''', (original_title, original_content, rewritten_article))

# Commit changes and close connection
conn.commit()
conn.close()

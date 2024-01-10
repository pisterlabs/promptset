import os
import psycopg2
from dotenv import load_dotenv
from openai import OpenAI

def generate_sql_query(question):
    load_dotenv()

    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    conn = psycopg2.connect(
        host="redash-postgres-1",
        database="youtube_data",
        user="postgres",
        password="postgres",
        port="5432"
    )

    cursor = conn.cursor()

    try:
        cursor.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'total_views_by_day'")
        schema_rows = cursor.fetchall()
        schema = {row[0]: row[1] for row in schema_rows}

        prompt = f"Retrieve data from the 'total_views_by_day' table based on the question: {question}."
        chat_completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        suggested_query = chat_completion.choices[0].message.content.strip()

        sql_query = suggested_query.split("\n\n")[1]

        return sql_query

    finally:
        cursor.close()
        conn.close()


question = "What is the total number of views?"
sql_query = generate_sql_query(question)
print(sql_query)
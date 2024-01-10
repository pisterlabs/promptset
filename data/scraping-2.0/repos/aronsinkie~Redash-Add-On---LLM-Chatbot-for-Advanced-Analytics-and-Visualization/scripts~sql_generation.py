import psycopg2
import openai

import os
from openai import OpenAI

def get_query(question):
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    conn = psycopg2.connect(
        host="localhost",
        database="youtube_data",
        user="postgres",
        password="postgres",
        port="15432"
    )

    cursor = conn.cursor()
    sql_query = None
    try:
        cursor.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'device_type_chart'")
        schema_rows = cursor.fetchall()
        schema = {row[0]: row[1] for row in schema_rows}

        prompt = f"Retrieve data from the 'device_type_chart' table based on the question: {question}."
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=50,
            n=1,
            stop=None,
            temperature=0.5,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )

        sql_query = response.choices[0].text.strip()

        cursor.execute(sql_query)
        result = cursor.fetchall()

    finally:
        cursor.close()
        conn.close()
    
    return sql_query
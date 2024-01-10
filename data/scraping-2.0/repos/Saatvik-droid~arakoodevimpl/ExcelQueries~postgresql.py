import os
import json

import psycopg2
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG")

conn = psycopg2.connect(database="test",
                        host="127.0.0.1",
                        user="postgres",
                        password="postgres",
                        port="5432")
cursor = conn.cursor()


def query_openai_chat_completion(messages, functions=None, function_call="auto"):
    if functions is None:
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-0613", messages=messages, temperature=0.7)
    else:
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-0613", messages=messages, temperature=0.7,
                                                  functions=functions, function_call=function_call)
    reply = completion.choices[0].message
    return reply


class Agent:
    @staticmethod
    def get_db_info():
        cursor.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_type='BASE TABLE';")
        tables = cursor.fetchall()
        tables = [table[0] for table in tables]
        table_schema = []
        samples = []
        for table in tables:
            cursor.execute(f"""
SELECT
  'CREATE TABLE ' || relname || E'\n(\n' ||
  array_to_string(
    array_agg(
      '    ' || column_name || ' ' ||  type || ' '|| not_null
    )
    , E',\n'
  ) || E'\n);\n'
from
(
  SELECT
    c.relname, a.attname AS column_name,
    pg_catalog.format_type(a.atttypid, a.atttypmod) as type,
    case
      when a.attnotnull
    then 'NOT NULL'
    else 'NULL'
    END as not_null
  FROM pg_class c,
   pg_attribute a,
   pg_type t
   WHERE c.relname = '{table}'
   AND a.attnum > 0
   AND a.attrelid = c.oid
   AND a.atttypid = t.oid
 ORDER BY a.attnum
) as tabledefinition
group by relname
;""")
            schema = cursor.fetchall()
            table_schema.append(schema)
            cursor.execute(f"SELECT * FROM {table} LIMIT 1")
            sample = cursor.fetchall()
            samples.append(sample[0])
        prompt = ""
        for t_s, s in zip(table_schema, samples):
            prompt += "\n" + t_s[0][0] + "SAMPLE ROW:\n"
            prompt += str(s)
        prompt += f"\nForiegn Keys:\n"
        cursor.execute(
            "SELECT tc.table_name, kcu.column_name, ccu.table_name AS foreign_table_name, ccu.column_name AS "
            "foreign_column_name FROM information_schema.table_constraints AS tc JOIN "
            "information_schema.key_column_usage AS kcu ON tc.constraint_name = kcu.constraint_name JOIN "
            "information_schema.constraint_column_usage AS ccu ON ccu.constraint_name = tc.constraint_name WHERE "
            "constraint_type = 'FOREIGN KEY';")
        fk_constraints = cursor.fetchall()
        for c in fk_constraints:
            prompt += f"Foreign Key in table {c[0]} in {c[1]} referencing table {c[2]} column {c[3]}\n"
        return prompt

    def run(self):
        prompt = """
You are working with a postgresql database. 
The programmer issues commands and you should translate them into SQL queries.

Human: {query}

```
{db_info}
```

Reply in the format:
{{
    "code": string
}}
        """
        db_info = self.get_db_info()
        query = input("Query:")
        prompt = prompt.format(query=query, db_info=db_info)
        messages = [
            {
                "role": "system", "content": prompt
            }
        ]
        print(prompt)
        reply = query_openai_chat_completion(messages).content
        print(reply)
        reply = json.loads(reply)["code"]
        cursor.execute(reply)
        print(cursor.fetchall())


if __name__ == "__main__":
    a = Agent()
    a.run()

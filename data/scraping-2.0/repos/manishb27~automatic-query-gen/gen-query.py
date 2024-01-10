import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")


def gen_query_from_user_input(db_name, table_name, case):
    prompt = """Create a query to get the data from the database and table given below. The query should return all the columns from the table and the rows that satisfy the given case.
    Database: """ + db_name + """
    Table: """ + table_name + """
    Case: """ + case + """
    Query:"""

    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        temperature=0.3,
        max_tokens=60,
        top_p=1,
        frequency_penalty=0.5,
        presence_penalty=0.5,
        stop=["\n"]
    )

    return response["choices"][0]["text"]


if __name__ == "__main__":
    db_name = input("Enter the database name: ")
    table_name = input("Enter the table name: ")
    case = input("Enter the case: ")

    print(gen_query_from_user_input(db_name, table_name, case))
import json
import openai
from db_chat.sql_builder.Query import Query
from db_chat.sql_builder.schema import Schema, validate_schema


def build_schema_prompt(schema: Schema):
    """
    Build the prompt for the LLM model
    """
    prompt = "Schema :\n"
    prompt += "Here are the list of tables and their columns:\n"
    for table in schema.tables.values():
        prompt += f"{table.name} - "
        for column in table.columns:
            prompt += f"{column.name},"

        prompt += "\n"
    return prompt


def build_rules():
    rules = """
    You will follow these rules while creating a json response:
    1. The json response will only contain the below listed keys
    2. `table` key will contain the name of the table to fetch the data from
    3. `fields` key will contain the list of fields to fetch from the chosen table. Fields of each table will be listed in a separate list after the rules.
    4. `filters` key will contain the list of filters to apply to the query. Each filter will have a 'field', 'operator' and 'value'. Allowed operators are 'eq', 'neq', 'gt', 'lt', 'gte', 'lte', 'like'
    5. `sort` -  will  have a 'field' and 'direction'.
        4.a. `field` - Field to sort to sort by. Use columns from the chosen `table` only . Do not make up new columns.
        4.b. `direction` - Allowed direction are 'asc' and 'desc'.
    6. `limit` key will contain the number of rows to limit the result to
    7. `offset` key will contain the number of rows to offset the result by

    """
    return rules


def build_prompt(schema: Schema, question: str):
    """
    Build the prompt for the LLM model
    """
    prompt = "You are an SQL Developer. Given a list of the tables and their columns of the database and a question from user. You will return a json response that will be used to fetch the data from the database.\n"

    prompt += build_rules()
    prompt += build_schema_prompt(schema)

    prompt += "Choose the fields that are found only in the respective tables. Don't choose the fields that are found in multiple tables. Don't make up columns of your own. \n"
    messages = []
    context = prompt
    messages.append(
        {
            "role": "system",
            "content": context,
        }
    )

    messages.append({"role": "user", "content": question})

    max_retry = 3
    retry_count = 0

    while retry_count < max_retry:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106", temperature=0, messages=messages, response_format={"type": "json_object"}
        )

        response_message = response.choices[0].message
        query = json.loads(response_message.content)
        query = Query(**query)
        print(query)

        validation_messages: list[str] = validate_schema(schema, query)

        if len(validation_messages) > 0:
            second_prompt = "The query you provided is not valid. Found the following problems."
            second_prompt += "\n".join(validation_messages)
            second_prompt += "Please try again following the rules mentioned.\n"

            print(validation_messages)
            messages.append({"role": "system", "content": ",".join(validation_messages)})

            retry_count += 1
            break
        else:
            break

    return query

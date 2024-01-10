from typing import Any, Dict, List

import openai
from sqlalchemy import Inspector

from core import env
from models import QueryResponseModel

from .flavor_queries import (
    COMMON_EXAMPLES,
    MSSQL_EXAMPLES,
    MYSQL_EXAMPLES,
    ORACLE_EXAMPLES,
    POSTGRES_EXAMPLES,
    SQLITE_EXAMPLES,
)

GPT_API_KEY = env.get("GPT_API_KEY")

if GPT_API_KEY is None:
    raise EnvironmentError("Missing environment variable: GPT_API_KEY")

openai.api_key = GPT_API_KEY


def fetch_tables(inspector: Inspector) -> List[str]:
    return inspector.get_table_names()


def fetch_primary_keys(table_name: str, inspector: Inspector) -> List[str]:
    indexes = inspector.get_indexes(table_name)
    for index in indexes:
        if index.get("primary_key"):
            return [col for col in index["column_names"] if col is not None]
    return []


def fetch_columns(table_name: str, inspector: Inspector) -> List[Dict[str, Any]]:
    columns_data = inspector.get_columns(table_name)
    primary_keys = fetch_primary_keys(table_name, inspector)
    fks = inspector.get_foreign_keys(table_name)

    columns = []
    for col in columns_data:
        column_description = {
            "name": col["name"],
            "type": col["type"].__visit_name__,
            "nullable": col["nullable"],
            "primary_key": col["name"] in primary_keys,
            "default": col.get("default"),
            "foreign_key": next(
                (fk for fk in fks if fk["constrained_columns"][0] == col["name"]), None
            ),
        }
        columns.append(column_description)

    return columns


def generate_sql_query(
    user_input: str,
    flavor: str,
    inspector: Inspector,
    conversation_history: List[Dict[str, str]],
) -> QueryResponseModel:
    flavor_examples = {
        "MySQL": MYSQL_EXAMPLES,
        "PostgreSQL": POSTGRES_EXAMPLES,
        "SQLite": SQLITE_EXAMPLES,
        "MSSQL": MSSQL_EXAMPLES,
        "Oracle": ORACLE_EXAMPLES,
    }

    if flavor not in flavor_examples:
        raise ValueError(f"Unsupported SQL flavor: {flavor}")

    # Fetch the database schema
    schema_description = get_database_schema(inspector)

    # extra prompt engineering
    extra = (
        "Do NOT use aliases in the query. Be explicit in the SQL syntax. "
        "When adding VARCHAR columns, always specify the length, like VARCHAR(255) "
        "But if the flavor is oracle, then it should be VARCHAR2(255 CHAR). "
        "Specify INNER JOIN when applicable. "
        "Remember to return an empty string if the user input is ANYTHING else besides instructions to make a query. "
        "This means you either output a query or an empty string with no exceptions at all. "
        "Don't include quotations when you return an empty string. "
        "This is the user input: "
    )

    # Construct the api_input
    api_input = f"[{flavor}] {schema_description} {extra} {user_input}"

    messages = (
        COMMON_EXAMPLES
        + flavor_examples[flavor]
        + conversation_history
        + [{"role": "user", "content": api_input}]
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )

    content = (
        response.choices[0].message["content"] or "I can only help with making queries."
    )
    return QueryResponseModel(engineered_input=api_input, sql_query=content)


def get_database_schema(inspector: Inspector) -> str:
    # Fetch all table names
    tables = fetch_tables(inspector)

    schema_description = []

    # Fetch columns for each table
    for table in tables:
        columns = fetch_columns(table, inspector)
        descriptions = []
        for col in columns:
            col_desc = f"{col['name']} ({col['type']})"
            if not col["nullable"]:
                col_desc += " NOT NULL"
            if col["default"]:
                col_desc += f" DEFAULT {col['default']}"
            if col["primary_key"]:
                col_desc += " PRIMARY KEY"
            if col["foreign_key"]:
                col["foreign_key"]
                col_desc += (
                    f" FOREIGN KEY REFERENCES {col['foreign_key']['referred_table']}"
                    f"({col['foreign_key']['referred_columns'][0]})"
                )
            descriptions.append(col_desc)

        schema_description.append(
            f"Table '{table}' has columns: {', '.join(descriptions)}."
        )

    return " ".join(schema_description)

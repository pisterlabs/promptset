import sqlite3
from langchain.tools import Tool
from ecommerce_agent.sql_query_schema.describe_tables_args_schema import DescribeTableArgsSchema

conn = sqlite3.connect('db.sqlite')


def describe_tables(table_names):
    cursor = conn.cursor()
    tables = ','.join("'" + table + "'" for table in table_names)
    table_details_rows = cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' and name IN ({tables});")
    return '\n'.join(table_detail_row[0] for table_detail_row in table_details_rows if table_detail_row[0] is not None)


describe_tables_tool = Tool.from_function(name="describe_tables",
                                          description="Given a list of table names, returns the schema of those tables.",
                                          func=describe_tables,
                                          args_schema=DescribeTableArgsSchema)

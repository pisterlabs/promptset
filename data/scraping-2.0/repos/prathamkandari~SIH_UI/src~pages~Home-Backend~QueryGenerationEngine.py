import openai
from extractKeyWords import extract_keywords
from connectToNeo4J import Neo4jConnection


uri = "neo4j+s://5206912c.databases.neo4j.io"
user = "neo4j"
password = "7-j3f1Gh_NS6k3k-Zzj_YuHtu7odMMlghS9i6gQ8J0o"
database_name = "neo4j" 

neo4j_conn = Neo4jConnection(uri, user, password)

def generateQuery(input_string, closest_nodes, neo4j_conn, openai_api_key):
    # Extracting only the string part (table names) from each tuple in closest_nodes
    table_names = [node[0] for node in closest_nodes]

    column_name_map = {}
    for table in table_names:
        columns = list(neo4j_conn.get_column_names(table))
        column_name_map[table] = columns


    example_queries = [

        "SELECT * FROM  [dbo].[Faculty.yaml] WHERE FacultyName = 'Neha Gupta'",
        "SELECT FacultyID, FacultyName, FacultyExperience FROM [dbo].[Faculty.yaml] WHERE FacultyExperience > 20",

    ]




    messages = [
        {"role": "system", "content": "You are a helpful assistant. Who Generates accurate SQL queries for users."},
        {"role": "user", "content": f"Input string: {input_string}\n\nTables and Columns:\n" + "\n".join([f"- Table {table}: {', '.join(columns)}" for table, columns in column_name_map.items()]) + "\n\nGenerate an SQL Server query based on the input string and the given tables and columns. Please keep it as accurate as possible." + "Example SQL Server queries based on a similar dataset:\n"+"\n".join([f"- {query}" for query in example_queries]) +f"Make sure to the query is of type [dbo]{table_names[0]}"+".yaml" + "type" +"Do not include closed square brackets for column names in the query"},
    ]

    openai.api_key = openai_api_key


    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )


    if response['choices'] and response['choices'][0]['message']:
        return response['choices'][0]['message']['content'].strip()
    else:
        return "No response generated."





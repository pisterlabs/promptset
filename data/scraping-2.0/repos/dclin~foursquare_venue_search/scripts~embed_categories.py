import snowflake.connector 
from snowflake.connector import DictCursor
import openai 

openai.api_key = "YOUR_OPENAI_API_KEY"

# Snowflake credential with update permission on the category_lookup table 
# The database and schema of where the category_lookup table is housed.
snowflake_credential = {
    'user': "YOUR_SNOWFLAKE_USER",
    'password': "YOUR_SNOWFLAKE_PASSWORD",
    'account': "YOUR_SNOWFLAKE_ACCOUNT",
    'warehouse': "YOUR_SNOWFLAKE_WAREHOUSE",
    'database': "YOUR_SNOWFLAKE_DATABASE",
    'schema': "YOUR_SNOWFLAKE_SCHEMA"
}

def get_embedding(category_str):
    try:
        response = openai.Embedding.create(
            input=category_str,
            model="text-embedding-ada-002"
        )
        embeddings = response['data'][0]['embedding']
        return embeddings 
    
    except Exception as e: 
        raise e 

def init_connection():
    return snowflake.connector.connect(**snowflake_credential)


# SQL assumes your Snowflake credential is in the database and schema of your category_lookup table
def get_all_categories(conn):
    sql = """SELECT category_id, category FROM category_lookup ORDER BY category_id"""

    categories = _run_query(conn,sql)

    return categories    

# SQL assumes your Snowflake credential is in the database and schema of your category_lookup table 
def update_embedding(conn, category_id, embeddings):
    sql = """
    UPDATE category_lookup 
    SET embedding='{0}'
    WHERE category_id = {1}
    """.format(embeddings, category_id)

    try:
        _run_query(conn, sql)
        return 1
    except:
        return 0 


def _run_query(conn, query_str):
    with conn.cursor(DictCursor) as cur:
        cur.execute(query_str)
        return cur.fetchall()


conn = init_connection()

categories = get_all_categories(conn)

for category in categories: 
    
    print(f"Embed category_id: {category['CATEGORY_ID']}")
    try:
        embeddings = get_embedding(category['CATEGORY'])
        print(f"Update category_id: {category['CATEGORY_ID']}")
        update_category = update_embedding(conn, category['CATEGORY_ID'], embeddings)
        if update_category == 1:
            print(f"Updated category_id: {category['CATEGORY_ID']}")
    except: 
        pass 


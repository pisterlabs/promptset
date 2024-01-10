
import json
from langchain.chat_models import ChatOpenAI

instruction = """
For the given question, determine the keywords for determining what tables and columns should be used to write a SQL query to answer the question.
"""

openai_api_key=''
def extract_entities(db, question,docs):
    extract_prompt = """
    Question: What are the top articles today?
    response: [{"table":"articles_daily"}, {"table":"articles"}]

    Question: Who are the top authors today?
    response: [{"table":"authors_daily"}, {"table":"authors"} ]

    Question: Who are the top authors today and what articles have they written?
    response: [{"table":"authors_daily"}, {"table":"authors"}, {"table":"articles"}, {"table":"articles_daily"} ]
    """

    prompt = (
        instruction
        + "return your response in the following format: "
        + extract_prompt
        + "Q: "
        + question
        + "\nresponse:"
    )

    llm = ChatOpenAI(
        temperature=0,
        model="gpt-3.5-turbo",
        openai_api_key=openai_api_key
    )

    response = llm.call_as_llm(prompt)
    response_json = json.loads(response)  # Convert the response to a JSON object
    documents = []
    processed_tables = set()

    for table in response_json:
        table_name = table['table']

        if table_name in processed_tables:
            continue

        processed_tables.add(table_name)

        # Find the document corresponding to the table_name
        table_doc = next((doc for doc in docs if doc.page_content == f"Table: {table_name}"), None)

        if table_doc is not None:
            table_meta = table_doc.metadata
            columns = list(table_meta.keys())
            docstring = f"Table {table_name}, columns = [*, {', '.join(columns)}]"
            documents.append(docstring)

    result = "\n".join(documents)
    return result

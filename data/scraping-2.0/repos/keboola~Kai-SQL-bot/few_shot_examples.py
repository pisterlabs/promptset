from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.agents.agent_toolkits import create_retriever_tool


few_shots = {'List all customers.': 'SELECT * FROM "customer";',
 'How many orders are there?': 'SELECT COUNT(*) FROM \"order\";',
 'Help me write a query to find the number of orders placed by each customer.': 'SELECT customer_id, COUNT(*) AS order_count FROM \"order\" GROUP BY customer_id;'
}

embeddings = OpenAIEmbeddings()

few_shot_docs = [Document(page_content=question, metadata={'sql_query': few_shots[question]}) for question in few_shots.keys()]
vector_db = FAISS.from_documents(few_shot_docs, embeddings)
retriever = vector_db.as_retriever()

tool_description = """
This tool will help you understand similar examples to adapt them to the user question.
Input to this tool should be the user question.
"""

retriever_tool = create_retriever_tool(
        retriever,
        name='sql_get_similar_examples',
        description=tool_description
    )
custom_tool_list = [retriever_tool, ]


custom_suffix = """
I should first get the similar examples I know.
If the examples are enough to construct the query, I can build it.
Otherwise, I can then look at the tables in the database to see what I can query.
Then I should query the schema of the most relevant tables
"""
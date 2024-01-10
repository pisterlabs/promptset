from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm

GOOGLE_API_KEY = 'Your API Key'

llm = GooglePalm(google_api_key=GOOGLE_API_KEY)
llm.temperature = 0.1

prompts = ["The opposite of hot is",'The opposite of cold is'] # according to the class prmpts must be in list
llm_result = llm._generate(prompts)

llm_result.generations[0][0].text
llm_result.generations[1][0].text


from langchain.utilities import SQLDatabase

db_user = 'root'
db_password = 'shiv'
db_host = 'localhost'
db_port = 3306
db_name = 'atliq_tshirts'

db = SQLDatabase.from_uri(f'mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}', sample_rows_in_table_info=3)
# print(db.table_info)

#sqlchain

from langchain_experimental.sql import SQLDatabaseChain

# db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
db_chain = SQLDatabaseChain.from_llm(llm, db)

qnsl = db_chain.run("if i sell all my LEVI's tshirt today how much revenue i will get?")

# print(qnsl)

# Few Shot Learning

few_shot = [
    {
        "Question": "total cost of inventory for small size tshirts?",
        "SQLQuery": "SELECT SUM(price * stock_quantity) FROM t_shirts WHERE size = 'S'",
        "SQLResult": "Result of the SQL query",
        "Answer": "9396"   
    },
]


from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

e = embeddings.embed_query('total cost of inventory for small size tshirts?')

# print(e)

to_vectorize = [" ".join(example.values()) for example in few_shot]

# print(to_vectorize)

#chromaDB

from langchain.vectorstores import Chroma

vectorstore = Chroma.from_texts(to_vectorize, embedding=embeddings, metadatas=few_shot)

# print(vectorstore)

from langchain.prompts import SemanticSimilarityExampleSelector

example_selector = SemanticSimilarityExampleSelector(
    vectorstore = vectorstore,
    k =2,
    )

# example_selector.select_examples({"Question": "total cost of inventory for small size tshirts?"})

print(example_selector.select_examples({"Question": "total cost of inventory for small size tshirts?"}))

from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt

# print(_mysql_prompt)
# print(PROMPT_SUFFIX)

from langchain.prompts.prompt import PromptTemplate

example_prompt = PromptTemplate(
    input_variables = ['Question', 'SQLQuery', 'SQLResult', 'Answer',],
    template = "\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}\n",
)

# print(example_prompt)

from langchain.prompts import FewShotPromptTemplate

few_shot_prompt = FewShotPromptTemplate(
    example_selector = example_selector,
    example_prompt = example_prompt,
    prefix = _mysql_prompt,
    suffix = PROMPT_SUFFIX,
    input_variables = ['input','table_info','top_k'],
)

# print(few_shot_prompt)

new_chain = SQLDatabaseChain.from_llm(llm, db, prompt=few_shot_prompt)

new_chain.run("total cost of inventory for large size tshirts?")
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from decouple import config
from langchain.chains import LLMChain
import coloredlogs, logging
logger = logging.getLogger(__name__)
coloredlogs.install(level=config('LOG_LEVEL', default='INFO'))

def build_query(question: str, yaml_files_for_tables: str, open_ai_model: str):
    logger.info("Building the query")
    llm = ChatOpenAI(model_name=open_ai_model, temperature=0)
    prompt = PromptTemplate(
        input_variables=["question", "yaml_files_for_tables"],
        template="""
        Write a snowflake SQL query to answer the following question: {question}
        (only retrun the snowflake SQL query, nothing else)

        Here are the yaml files of the tables you have access to:
        {yaml_files_for_tables}
        """,
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    query = chain.run({"question": question, "yaml_files_for_tables": yaml_files_for_tables})
    logger.info(f"Query: {query}")
    return query
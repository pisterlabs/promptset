from langchain import PromptTemplate
from langchain.output_parsers.list import CommaSeparatedListOutputParser
from langchain.prompts import load_prompt

_DEFAULT_TEMPLATE = """You are a SnowflakeDB expert. Given an input question, first create a syntactically correct SnowflakeDB query to run, 
then look at the results of the query and return the answer. Pay attention to use only the column names you can see in the tables below

Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLValidation: "Have you generated correct output?"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

Only use the following tables:
{table_info}

Ensure conditions while generating the query
- Prefer JOIN over Subquery
- Prefer = over Like for string matching
- Always wrap date inside TRY_TO_TIMESTAMP function
- Generate the query by avoiding all sqlalchemy errors
- Avoid all sqlalchemy.exc.ProgrammingErrors

Example of casting date
```
SELECT TRY_TO_TIMESTAMP('2015-01-01');
```

Example of WRONG Subquery
```
SELECT COUNT(DISTINCT person_id) FROM xyz 
WHERE city_names = (SELECT city_names FROM concept WHERE address_city = 'Kolkata')
```

If the query returns any Error, REGENERATE the query and RERUN

Question: {input}"""

_DECIDER_TEMPLATE = """Given the below input question and list of specific tables, output a comma separated list of the table names that are necessary to answer this question.

Question: {query}

Table Names: {table_names}

Relevant Table Names:"""

test = "abc"
temp = test+"iven the below input question and list of specific tables, {query} output a comma separated list of the table names that are necessary to answer this question.Question"

def generate_vectordb_template(response) -> PromptTemplate:
    VECTORDB_TEMPLATE = f"""Consider the information extracted.
    {response}\n\n"""
    template =  VECTORDB_TEMPLATE + "For the given question, the list of tables that can be used are {table_names}"
    return PromptTemplate(
        input_variables=["table_names"],
        template=template,
        output_parser=CommaSeparatedListOutputParser(),
    )


def get_prompt_template(db_response) -> PromptTemplate:
    return PromptTemplate(
        input_variables=["input","table_info"], template=_DEFAULT_TEMPLATE
    )

SNOWFLAKE_DECIDER_PROMPT = PromptTemplate(
    input_variables=["query","table_names"],
    template=_DECIDER_TEMPLATE,
    output_parser=CommaSeparatedListOutputParser(),
)
import os
import re

import chainlit as cl
import openai
import pandas as pd
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from llama_index import LLMPredictor, ServiceContext
from llama_index.callbacks.base import CallbackManager
from llama_index.query_engine import PandasQueryEngine
from pandasql import sqldf

load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")


df = pd.read_csv("./ronaldo_all_match_results.csv")

RAW_INSRUCTION = (
    "We wish to convert this query to executable SQL code using SQL.\n"
    # "The final line of code should be a Python expression that can be called "
    # "with the `eval()` function. This expression should represent a solution "
    # "to the query. This expression should not have leading or trailing "
    # "quotes.\n"
)

CUSTOM_INSTRUCTION = f"""This dataframe contains all the stats of all club goals of Cristiano Ronaldo (CR7).
- Season: The soccer season when the goal was scored (02/03, likely 2002-2003).
- Competition: The league or competition where the goal was scored (Liga Portugal).
- Matchday: The matchday during the season (6th match).
- Date: The date the match took place (10-07-02, October 7, 2002).
- Venue: Whether the match was at home (H) or away (A).
- Club: The team Ronaldo played for (Sporting CP).
- Opponent: The opposing team (Moreirense FC).
- Result: The final score of the match (3:00, Sporting CP won 3-0).
- Playing_Position: Ronaldo's position (LW, Left Wing).
- Minute: The minute when Ronaldo scored (34th minute).
- At_score: The score when Ronaldo scored (2:00, Sporting CP led 2-0).
- Assist_by: How the goal was assisted (Solo run, no direct assist)

{RAW_INSRUCTION}

"""


def pysqldf(q):
    return sqldf(q, globals())


def remove_sql_code_quotes(sql_code):
    # Define the regular expression pattern to match SQL code blocks
    pattern = r"```sql\n(.*?)\n```|`[^`]+`"

    # Use re.sub to remove code quotes
    cleaned_sql = re.sub(
        pattern,
        lambda match: match.group(1) if match.group(1) else "",
        sql_code,
        flags=re.DOTALL,
    )

    return cleaned_sql


def custom_output_processor(
    output: str, df: pd.DataFrame, **output_kwargs
) -> str:
    clean_output = remove_sql_code_quotes(output)

    result = pysqldf(clean_output)

    cl.user_session.set("chat_result_df", df)

    return f"Using Query:\n```sql\n{output}\n```\n Result:\n\n {result}"


llm_predictor = LLMPredictor(
    llm=ChatOpenAI(
        temperature=0,
        model_name="gpt-4",
        streaming=True,
    ),
)

service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor,
    chunk_size=512,
)
query_engine = PandasQueryEngine(
    df,
    service_context=service_context,
    instruction_str=CUSTOM_INSTRUCTION,
    output_processor=custom_output_processor,
)

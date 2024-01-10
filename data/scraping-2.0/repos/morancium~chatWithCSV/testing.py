import tabula as tb
import pandas as pd
from thefuzz import fuzz
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
from dotenv import load_dotenv
from utils import cleaning_df, load_doc
import os
import re
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
llm = OpenAI(api_token=OPENAI_API_KEY,temperature=0)
pandas_ai = PandasAI(llm=llm)
# main_frame=cleaning_df("statement_unlocked_may.pdf")
# main_frame.to_csv("out.pdf", sep=',', index=False, encoding='utf-8')
def load_docs(path: str):
    dfs = tb.read_pdf(path, pages="all", stream=True)
    return dfs
dfs=load_docs("pdfs\statement_unlocked_may.pdf")
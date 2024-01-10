
import os
import openai
import pandas as pd
import sqlalchemy
from decouple import config

OPENAI_API_KEY = config('OPENAI_API_KEY')

openai.api_key = os.getenv(OPENAI_API_KEY)

# print(openai.api_key)

df = pd.read_csv("data/sales_data_sample.csv")
df.groupby("QTR_ID").sum()['SALES']


# Temp DG in RAM
# PUSH Pandas DF -> TEMP DB
# SQL query on Temp DB

from sqlalchemy import text
from sqlalchemy import create_engine

temp_db = create_engine('sqlite:///:memory:', echo=True)

data = df.to_sql(name='Sales', con=temp_db)


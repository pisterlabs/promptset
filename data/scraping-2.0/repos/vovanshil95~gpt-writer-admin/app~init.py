from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import openai

from config import sqlalchemy_url, OPENAI_API_KEY

sql_engine = create_engine(sqlalchemy_url)
sqlalchemy_session = sessionmaker(sql_engine)

openai.api_key = OPENAI_API_KEY
import flask_praetorian
import flask_sqlalchemy
import boto3
import openai
import os
from newsapi import NewsApiClient

guard = flask_praetorian.Praetorian()
db = flask_sqlalchemy.SQLAlchemy()
storage_client = boto3.client(
    "s3",
    region_name="us-east-2",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
)
openai.api_key = os.getenv("OPENAI_API_KEY")
news_client = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))

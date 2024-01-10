# Databricks notebook source
import logging
import time
import datetime
from typing import List

from pyspark.sql.functions import col

from config.integration_config import AWSConfig
from config.custom_logging import setup_logging

# import torch
# from transformers import pipeline as tpipeline
from pyspark.sql. functions import collect_list, concat_ws, col, count, sum as pysum
import pandas as pd
from pyspark.sql.functions import pandas_udf

# COMMAND ----------

# OpenAi token
openai_token = dbutils.secrets.get(scope="openai_credentials", key="api.token")

# AWS credentials
aws_access_key = dbutils.secrets.get(scope="aws_credentials", key="data_services.access_key")
aws_secret_key = dbutils.secrets.get(scope="aws_credentials", key="data_services.secret_key")

sc._jsc.hadoopConfiguration().set("fs.s3a.access.key", aws_access_key)
sc._jsc.hadoopConfiguration().set("fs.s3a.secret.key", aws_secret_key)

aws_config = AWSConfig(aws_access_key_id=aws_access_key, aws_secret_key=aws_secret_key)
boto3_config = aws_config.create_boto3_session()

# == S3 config
path_data_bucket = "neurum-ai-factored-datathon"
path_silver_amz_reviews = f"s3a://{path_data_bucket}/silver/amazon/reviews"
path_gold_amz_reviews = f"s3a://{path_data_bucket}/gold/amazon/reviews"
path_gold_amz_reviews_output = f"s3a://{path_data_bucket}/gold/amazon/llm_reviews"

path_llm_models_bucket = "trird-party-llm-factored"
path_llama_model = f"s3a://{path_llm_models_bucket}"
folder_llama_model = f"s3a://{path_llm_models_bucket}/llama_model/"
folder_llama_tokenizer = f"s3a://{path_llm_models_bucket}/llama_tokenizer/"

# == Mount S3 and Databricks
db_mount_name = "/dbfs/mnt/llama_model"
# dbutils.fs.mounts()
# dbutils.fs.mount(path_llama_model, db_mount_name[5:])
# dbutils.fs.unmount(db_mount_name[5:])

# COMMAND ----------

current_date = datetime.datetime.utcnow()
one_day = datetime.timedelta(days=1)
previous_date = current_date - one_day
default_run_date = previous_date.strftime("%Y-%m-%d")

# Widgets
dbutils.widgets.text("processing_date", default_run_date, "processing_date")

processing_date = (
    dbutils.widgets.get("processing_date")
    if (dbutils.widgets.get("processing_date") != "")
    else default_run_date
)

year_month = processing_date[:-1] + "1"

print(processing_date, year_month)

# COMMAND ----------

df_gold_reviews = spark.read.format("delta").load(path_gold_amz_reviews).filter(col("year_month") == year_month).filter(col("date") == processing_date).filter(col("review_text").isNotNull()).dropDuplicates()#.filter(col("asin").isin("B016M7XUY2", "B016IX1736"))
print(df_gold_reviews.count())
display(df_gold_reviews)#.groupBy("asin").count().orderBy(col("count").desc()))

# COMMAND ----------

# DBTITLE 1,The pandas_udf don't accept request_limit decorator or backoff, so we made a manual request speed.
import pandas as pd
import openai
from pyspark.sql.functions import pandas_udf
import time

DEFAULT_SYSTEM_PROMPT = """You are an API response, just provide a valid JSON response from a review input:
{"sentiment": string # Extract the sentiment, in 1 label word, just select one of the next options: [Neutral, Positive, Negative], don't use a different label.
"actionable": string # Advice of actionable to the product selector of an e-commerce based on the review (1 actionable in just 3 words).
"keywords": string # 3 Keyword extraction (3 keywords) as a string with comma-separated values. Avoid keywords like names, dates, locations, and organizations/companies.
"response": string # Short response to the review kindly and thanking the feedback to improve products offered in just a few words.}"""

CALLS_PER_MINUTE = 3500 # 3500 RPM
SECONDS_BETWEEN_CALLS = 60.0 / CALLS_PER_MINUTE
global total_tokens_used
total_tokens_used = 0

@pandas_udf('string')
def analyze_review(reviews: pd.Series) -> pd.Series:
    sentiments = []

    for review in reviews:
        openai.api_key = openai_token
        prompt_content = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": review
            }
        ]
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k-0613",
            messages=prompt_content,
            temperature=1,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        
        global total_tokens_used
        total_tokens_used += response['usage']['total_tokens']

        # Aquí puedes extraer la información relevante de la respuesta
        sentiment = response.choices[0].message.content
        sentiments.append(sentiment)
        
        time.sleep(SECONDS_BETWEEN_CALLS)

    return pd.Series(sentiments)

review_by_product_df = df_gold_reviews.withColumn("sentiment", analyze_review(df_gold_reviews["review_text"]))

# COMMAND ----------

display(review_by_product_df)

# COMMAND ----------

from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType

# Definición del esquema del JSON
schema = StructType([
    StructField("sentiment", StringType(), True),
    StructField("actionable", StringType(), True),
    StructField("keywords", StringType(), True),
    StructField("response", StringType(), True)
])

# Supongamos que tienes un DataFrame llamado 'df_silver_reviews' y la columna con la cadena JSON se llama 'json_column'
json_column = "review_insight"

# Parsear la columna JSON y crear nuevas columnas para cada clave
result_df = review_by_product_df \
    .withColumnRenamed("sentiment", json_column) \
    .withColumn("parsed", from_json(col(json_column), schema)) \
    .select("*",
            col("parsed.sentiment").alias("sentiment"),
            col("parsed.actionable").alias("actionable"),
            col("parsed.keywords").alias("keywords"),
            col("parsed.response").alias("response")) \
    .drop("parsed", json_column) # Si quieres, puedes eliminar la columna "parsed"

# Ahora 'result_df' tiene nuevas columnas para cada una de las claves del JSON

# COMMAND ----------

display(result_df)

# COMMAND ----------

(
    result_df.write.format("delta")
    .mode("overwrite")
    .partitionBy("date")
    .option("replaceWhere", f"date = '{processing_date}'")
    .save(path_gold_amz_reviews_output)
 )

# COMMAND ----------

df_gold_llm = spark.read.format("delta").load(path_gold_amz_reviews_output)
display(df_gold_llm)

# COMMAND ----------

df_gold_llm.write.format("delta").mode("append").saveAsTable("factored.amazon.gold_llm_review")

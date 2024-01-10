# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # 01 DATA PREPARATION
# MAGIC
# MAGIC In this notebook we will be preparing our data. 
# MAGIC
# MAGIC Before running this code, you will need to download the [CMU Movie Dataset](http://www.cs.cmu.edu/~ark/personas/) and store the files in a Databricks Volume. 

# COMMAND ----------

# MAGIC %run ./resources/variables

# COMMAND ----------

import huggingface_hub
hf_token = dbutils.secrets.get(f"{secrets_scope}", f"{secrets_hf_key_name}")
from huggingface_hub import login
login(token=hf_token)

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import types as T
from databricks.sdk.runtime import *


from langchain.text_splitter import TokenTextSplitter
from typing import Iterator
import pandas as pd
from pyspark.sql.functions import rand,when

from transformers import LlamaTokenizer
from typing import Iterator, List, Dict
import pandas as pd
from random import randint


# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## 1. BRONZE DATA LAYER
# MAGIC
# MAGIC
# MAGIC We will be ingesting the files movie.metadata.tsv and plot_summaries.txt

# COMMAND ----------

#function to read movie.metadata.tsv
def ms_movie_metadata_bronze(volume_path):
    return (
        spark.read.format("csv")
        .option("delimiter", "\t")
        .option("inferSchema", "true")
        .option("header", "false")
        .load(f"{volume_path}/movie.metadata.tsv")
        .toDF(
            "wikipedia_movie_id",
            "freebase_movie_id",
            "movie_name",
            "movie_release_date",
            "movie_box_office_revenue",
            "movie_runtime",
            "movie_languages",
            "movie_countries",
            "movie_genres"))

# COMMAND ----------

#function to read plot_summaries.txt
def ms_plot_summaries_bronze(volume_path):
    return (
        spark.read.format("csv")
        .option("delimiter", "\t")
        .option("inferSchema", "true")
        .option("header", "false")
        .load(f"{volume_path}/plot_summaries.txt")
        .toDF(
            "wikipedia_movie_id",
            "plot_summary"))

# COMMAND ----------

#write movie metadata to delta table
df = ms_movie_metadata_bronze(volume_path)
df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.ms_movie_metadata_bronze")
display(spark.table(f"{catalog}.{schema}.ms_movie_metadata_bronze"))

# COMMAND ----------

#write plot summaries to delta table
df = ms_plot_summaries_bronze(volume_path)
df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.ms_plot_summaries_bronze")
display(spark.table(f"{catalog}.{schema}.ms_plot_summaries_bronze"))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 2. SILVER DATA LAYER
# MAGIC
# MAGIC Joining movie metadata with plot summaries and some data cleanup.

# COMMAND ----------

#reading metadata table and some data cleanup
def read_movie_metadata(catalog, schema):
    return (
        spark.table(f"{catalog}.{schema}.ms_movie_metadata_bronze")
        .withColumn("movie_countries", F.from_json("movie_countries", "map<string,string>"))
        .withColumn("movie_countries", F.map_values("movie_countries"))
        .withColumn("movie_languages", F.from_json("movie_languages", "map<string,string>"))
        .withColumn("movie_languages", F.map_values("movie_languages"))
        .withColumn("movie_genres", F.from_json("movie_genres", "map<string,string>"))
        .withColumn("movie_genres", F.map_values("movie_genres")))

#reading plot summaries table
def read_plot_summaries(catalog, schema):
    return spark.table(f"{catalog}.{schema}.ms_plot_summaries_bronze")

#joining plot summaries with metadata tables
def read_movie_documents(catalog, schema):
    return (
        read_movie_metadata(catalog, schema)
        .join(read_plot_summaries(catalog, schema), "wikipedia_movie_id")
        .withColumn("document", F.concat_ws(
            "\n\n",
            F.concat_ws(" ", F.lit("movie name:"), F.col("movie_name")),
            F.concat_ws(" ", F.lit("plot summary:"), F.col("plot_summary")),
            F.concat_ws(" ", F.lit("genres:"), F.concat_ws(", ", F.col("movie_genres"))))))

# COMMAND ----------

documents = read_movie_documents(catalog, schema)

# COMMAND ----------

# adding a column for profile_type (random assignment)
documents=documents.withColumn('childproof', when(rand() > 0.95, 1).otherwise(0))
documents=documents.withColumn('premium', when(rand() > 0.70, 1).otherwise(0))

# COMMAND ----------

# adding for rating (random assignment)
def rating_generator():
  return randint(50,100)

rating_generator_udf = F.udf(lambda: rating_generator(), T.IntegerType())

documents=documents.withColumn('rating', rating_generator_udf())

# COMMAND ----------

print((documents.count(), len(documents.columns)))

# COMMAND ----------

display(documents)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### 2A. DATA EXPLORATORY ANALYSIS
# MAGIC
# MAGIC Let's explore the data and determine the average number of tokens per document. This is important to understand because LLMs have token input limits; and in this RAGs architecture we will be passing plot summaries as context. Because we are going to be using [Llama-2-7b-chat from hugging face](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), we should use the tokenizer used in that model. You can access all Llama modesl through Hugging Face through [this](https://huggingface.co/meta-llama) link. 

# COMMAND ----------

tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')

# COMMAND ----------

barbie_example = """
To live in Barbie Land is to be a perfect being in a perfect place. Unless you have a full-on existential crisis. Or you’re a Ken.

From Oscar-nominated writer/director Greta Gerwig (“Little Women,” “Lady Bird”) comes “Barbie,” starring Oscar-nominees Margot Robbie (“Bombshell,” “I, Tonya”) and Ryan Gosling (“La La Land,” “Half Nelson”) as Barbie and Ken, alongside America Ferrera (“End of Watch,” the “How to Train Your Dragon” films), Kate McKinnon (“Bombshell,” “Yesterday”), Michael Cera (“Scott Pilgrim vs. the World,” “Juno”), Ariana Greenblatt (“Avengers: Infinity War,” “65”), Issa Rae (“The Photograph,” “Insecure”), Rhea Perlman (“I’ll See You in My Dreams,” “Matilda”), and Will Ferrell (the “Anchorman” films, “Talladega Nights”). The film also stars Ana Cruz Kayne (“Little Women”), Emma Mackey (“Emily,” “Sex Education”), Hari Nef (“Assassination Nation,” “Transparent”), Alexandra Shipp (the “X-Men” films), Kingsley Ben-Adir (“One Night in Miami,” “Peaky Blinders”), Simu Liu (“Shang-Chi and the Legend of the Ten Rings”), Ncuti Gatwa (“Sex Education”), Scott Evans (“Grace and Frankie”), Jamie Demetriou (“Cruella”), Connor Swindells (“Sex Education,” “Emma.”), Sharon Rooney (“Dumbo,” “Jerk”), Nicola Coughlan (“Bridgerton,” “Derry Girls”), Ritu Arya (“The Umbrella Academy”), Grammy Award-winning singer/songwriter Dua Lipa and Oscar-winner Helen Mirren (“The Queen”).

Gerwig directed “Barbie” from a screenplay by Gerwig & Oscar nominee Noah Baumbach (“Marriage Story,” “The Squid and the Whale”), based on Barbie by Mattel. The film’s producers are Oscar nominee David Heyman (“Marriage Story,” “Gravity”), Robbie, Tom Ackerley and Robbie Brenner, with Michael Sharp, Josey McNamara, Ynon Kreiz, Courtenay Valenti, Toby Emmerich and Cate Adams serving as executive producers.

Gerwig’s creative team behind the camera included Oscar-nominated director of photography Rodrigo Prieto (“The Irishman,” “Silence,” “Brokeback Mountain”), six-time Oscar-nominated production designer Sarah Greenwood (“Beauty and the Beast,” “Anna Karenina”), editor Nick Houy (“Little Women,” “Lady Bird”), Oscar-winning costume designer Jacqueline Durran (“Little Women,” “Anna Karenina”), visual effects supervisor Glen Pratt (“Paddington 2,” “Beauty and the Beast”), music supervisor George Drakoulias (“White Noise,” “Marriage Story”) and Oscar-winning composer Alexandre Desplat (“The Shape of Water,” “The Grand Budapest Hotel”).

Warner Bros. Pictures Presents a Heyday Films Production, a LuckyChap Entertainment Production, a Mattel Production, “Barbie.” The film will be distributed worldwide by Warner Bros. Pictures and released in theaters only nationwide on July 21, 2023 and beginning internationally on July 19, 2023.
"""

# COMMAND ----------

print(f"length of Barbie about page: {len(tokenizer.encode(barbie_example))}")

# COMMAND ----------

# UDF to determine the number of tokens using the llama-2-7b tokenizer

@F.pandas_udf("long")
def num_tokens_llama(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    login(token=hf_token)
    tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
    try:
        for x in batch_iter:
            yield x.apply(lambda s: len(tokenizer.encode(s)))
    finally:
        pass

# COMMAND ----------

documents = (
    documents
    .withColumn("document_num_chars", F.length("document"))
    .withColumn("document_num_words", F.size(F.split("document", "\\s")))
    .withColumn("document_num_tokens_llama", num_tokens_llama("document"))
)

# COMMAND ----------

display(documents)

# COMMAND ----------

documents.createOrReplaceTempView("documents")

# COMMAND ----------

# MAGIC %sql
# MAGIC select 
# MAGIC   avg(document_num_tokens_llama) as mean_tokens,
# MAGIC   max(document_num_tokens_llama) as max_tokens,
# MAGIC   min(document_num_tokens_llama) as min_tokens,
# MAGIC   sum(case when document_num_tokens_llama>3500 then 1 else 0 end) as documents_3500
# MAGIC from documents

# COMMAND ----------

#to keep things simple for this workshop we are going to remove all documents with a token limit about 3500. This is because Llama-2 has a token input limit of 4096 tokens. 

documents=documents.filter(documents.document_num_tokens_llama <=3500)

# COMMAND ----------

#write to delta table
documents.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.movie_documents_silver")

# COMMAND ----------

# delta table to use
df=spark.sql(f'''select wikipedia_movie_id, document, movie_name, movie_release_date, movie_runtime, childproof, premium, rating, document_num_tokens_llama, document_num_chars, 
document_num_words from {catalog}.{schema}.movie_documents_silver limit 10000;''')

# COMMAND ----------

#creating a subset of data for vector search delta sync
df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.{sync_table_name}")

# COMMAND ----------

spark.sql(f'''
          ALTER TABLE {catalog}.{schema}.movie_documents_for_sync SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
          ''')

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 3. DATA CHUNKING
# MAGIC
# MAGIC We won't be using chunking in this rag bot, but I wanted to include how you would do this. This is a good strategy if you need extra control over token input. 

# COMMAND ----------

print(f"chunk_size: {chunk_size}")
print(f"chunk_overlap: {chunk_overlap}")

# COMMAND ----------

def split_documents(dfs: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    fn = lambda s: text_splitter.split_text(s)
    for df in dfs:
        df.loc[:, "text"] = df.loc[:, "plot_summary"].apply(fn)
        df = df.loc[:, ["wikipedia_movie_id", "text"]]
        df = df.explode("text").reset_index().rename(columns={'index' : 'chunk_index'})
        df['chunk_index'] = df.groupby('chunk_index').cumcount()
        yield df.loc[:, ["wikipedia_movie_id", "chunk_index", "text"]]

# COMMAND ----------

movie_df=spark.table(f"{catalog}.{schema}.movie_documents_silver")

metadata_df = (
    movie_df.select([
        "wikipedia_movie_id", 
        "movie_name",
        "movie_release_date",
        "movie_runtime",
        "childproof",
        "premium",
        "rating",
        "movie_languages",
        "movie_genres",
        "movie_countries",
        "document_num_tokens_llama", 
        "document_num_chars", 
        "document_num_words",
        "document"]))

# COMMAND ----------

results_schema = T.StructType([
    T.StructField("wikipedia_movie_id", T.IntegerType()),
    T.StructField("chunk_index", T.LongType()),
    T.StructField("text", T.StringType())])


results = (
    movie_df.mapInPandas(split_documents, results_schema)
    .withColumn("id", F.concat_ws("_", 
        F.col("wikipedia_movie_id").cast("string"), 
        F.col("chunk_index").cast("string")))
    .join(metadata_df, "wikipedia_movie_id"))

# COMMAND ----------

display(results)

# COMMAND ----------

results.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.movie_documents_silver_chunked")

# COMMAND ----------



# Databricks notebook source
# MAGIC %md 
# MAGIC Need to create
# MAGIC - cjc_cap_markets catalog 
# MAGIC - capm_data schema
# MAGIC - 10kreports volume

# COMMAND ----------

# MAGIC %md `databricks secrets put --scope tokens --key canadaeh-openaikey --string-value`

# COMMAND ----------

# MAGIC
# MAGIC %md 
# MAGIC needed to upload a pdf 
# MAGIC `file_path2 = '/Volumes/cjc_cap_markets/capm_data/10kreports/boa-2022-10k.pdf'`
# MAGIC

# COMMAND ----------

# MAGIC %md ### OpenAI credentials
# MAGIC ```
# MAGIC import openai
# MAGIC
# MAGIC openaikey = dbutils.secrets.get("tokens", "canadaeh-openaikey")
# MAGIC openai.api_key = openaikey
# MAGIC openai.api_type = "azure"
# MAGIC openai.api_base = "https://canada-eh-openai.openai.azure.com/"
# MAGIC openai.api_version = "2023-07-01-preview"
# MAGIC aoai_model = "gpt-35-deployment"
# MAGIC ```

# COMMAND ----------



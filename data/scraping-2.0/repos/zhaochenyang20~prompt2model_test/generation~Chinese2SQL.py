from prompt2model.prompt_parser import OpenAIInstructionParser, TaskType
import os
import openai
from prompt2model.dataset_generator.base import DatasetSplit
from prompt2model.dataset_generator.openai_gpt import OpenAIDatasetGenerator
import logging

openai.api_key = os.environ["OPENAI_API_KEY"]
logging.basicConfig(level=logging.INFO)

prompt = """Chinese2SQL is an NLP task that involves converting natural language queries written in Chinese into SQL queries for querying relational databases.

For this task, the input is a Chinese string that describes a natural language query. The output is the corresponding SQL query.

Here are some examples:

input="北京市的人口是多少？"
output="SELECT population FROM cities WHERE city_name = '北京市'"

input="查询销售额大于10000的产品。"
output="SELECT * FROM products WHERE sales > 10000"

input="显示2019年至今的订单数量。"
output="SELECT COUNT(*) FROM orders WHERE order_date >= '2019-01-01'"
"""
prompt_spec = OpenAIInstructionParser(task_type=TaskType.TEXT_GENERATION)
prompt_spec.parse_from_prompt(prompt)
unlimited_dataset_generator = OpenAIDatasetGenerator(temperature=1.7, batch_size=5)
unlimited_dataset_generator.generate_dataset_split(
    prompt_spec, 200, split=DatasetSplit.TRAIN
)

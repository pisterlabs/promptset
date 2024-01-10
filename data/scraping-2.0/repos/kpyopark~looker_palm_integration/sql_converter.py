from abc import *
from enum import Enum
from typing import Callable
from parsors import parse_json_response, parse_python_object
import vertexai
from langchain.chat_models import ChatVertexAI
from langchain.llms import VertexAI
import os
from langchain.embeddings import VertexAIEmbeddings
from google.cloud import bigquery
from vector_util import VectorDatabase
from lookml_palm import LookMaker


PREPARED_STATEMENT_PARAMETER_CHAR_BIGQUERY = '?'
PREPARED_STATEMENT_PARAMETER_CHAR_OTHERS = '%s'
PREPARED_STATEMENT_PARAMETER_CHAR = PREPARED_STATEMENT_PARAMETER_CHAR_BIGQUERY


PROJECT_ID = os.getenv("PROJECT_ID")  # @param {type:"string"}
vertexai.init(project=PROJECT_ID, location="us-central1")

llm_vertex = VertexAI(
    #model_name="text-bison@latest",
    model_name="text-bison-32k",
    max_output_tokens=8000,
    temperature=0,
    top_p=0.8,
    top_k=40,
)

# TODO : Change below routine before to use in production

llm = llm_vertex
embeddings = VertexAIEmbeddings()
client = bigquery.Client()
sample_dataset_id = 'bigquery-public-data.thelook_ecommerce'

## TODO

class SqlConverterEventType(Enum):
  INIT = 0
  PROGRESSING = 1
  FINISHED = 2
  NEED_MORE_INFO = 3
  ERROR = 5
  VALIDATING = 6

class SqlConverterResultType(Enum):
  SQL = 0,
  RESULT_SET = 1,
  DASHBOARD = 2,
  LOOK = 3

class SqlConverterResult:
  def __init__(self, result_type : SqlConverterResultType, converted_sql : str, result_set : any, dashboard_url : str, look_url : str):
    self.result_type = result_type
    self.result_set = result_set
    self.converted_sql = converted_sql
    self.dashboard_url = dashboard_url
    self.look_url = look_url
  
  def get_converted_sql(self):
    return self.converted_sql
  
  def get_look_url(self):
    return self.look_url
  
  def get_dashboard_url(self):
    return self.dashboard_url
  
  def get_result_set(self):
    return self.result_set

conversation_maps = {}

def get_sql_converter(conversation_id, question):
  if conversation_id not in conversation_maps:
    conversation_maps[conversation_id] = SqlConverterFactory(question).get_sql_converter()
  return conversation_maps[conversation_id]

class SqlConverterFactory:

  def __init__(self, question):
    self.question = question
  
  def get_sql_converter(self):
    return DirectSqlConverter()


class SqlCrawler(ABC):
    def __init__(self, properties):
      self.properties = properties
      self.vdb = VectorDatabase()
    
    def truncate_table(self):
      if self.vdb is not None:
        self.vdb.truncate_table()

    @abstractmethod
    def crawl(self):
      pass

class BigQuerySchemaScrawler(SqlCrawler):

  # properties = {'dataset_id' : 'bigquery-public-data.thelook_ecommerce'}
  def __init__(self, properties):
    super().__init__(properties)

  def crawl(self):
    table_schemas = self.crawl_table_schemas()
    enriched_table_schemas = self.enrich_table_schemas(table_schemas)
    self.write_schema_to_vdb(enriched_table_schemas)

  def crawl_table_schemas(self):
    dataset_id = self.properties['dataset_id']
    tables = client.list_tables(dataset_id)
    table_schemas = []
    for table in tables:
      table_id = f"{dataset_id}.{table.table_id}"
      table_schema = client.get_table(table).schema
      table_schemas.append({'table_id': table_id, 'table_schema': table_schema})
    return table_schemas

  def enrich_schema_information(self, table_name, table_schema):
    sample_json = """
    {
      "table_name" : "bigquery-public-data.thelook_ecommerce.orders",
      "table_description" : "Orders placed by customers on the Look, an online store that sells clothing, shoes, and other items.",
      "columns" : [ 
        { 
          "column_name" : "order_id",
          "column_description" : "A unique identifier for the order. This is populated when an order is created.", 
          "column_type" : "INT64"
        }
      ]
    }
    """
    prompt_template = """You are a Looker Developer, enrich the schama information for the table {table_name} with the following information:

    table_name : 
    {table_name}

    table_column_schema :
    {table_column_schema}

    output_json :
    {sample_json}
    """
    prompt = prompt_template.format(table_name=table_name, table_column_schema=table_schema, sample_json=sample_json)
    response = llm.predict(prompt)
    return response

  def enrich_table_schemas(self, table_schemas):
    results = []
    for table_schema in table_schemas:
      table_name = table_schema['table_id']
      one_table_schema = table_schema['table_schema']
      response = self.enrich_schema_information(table_name, one_table_schema)
      results.append(parse_json_response(response))
    return results

  def write_schema_to_vdb(self, enriched_table_schemas):
    if enriched_table_schemas is None or len(enriched_table_schemas) == 0:
      return
    for enriched_table_schema in enriched_table_schemas:
      description = enriched_table_schema['table_description']
      desc_vector = embeddings.embed_query(description)
      self.vdb.insert_record(sql=None, parameters=None, description=description, explore_view=None, model_name=None, table_name=str(enriched_table_schema['table_name']), column_schema=str(enriched_table_schema['columns']), desc_vector=desc_vector)


class LookerNavExplorerCrawer(SqlCrawler):

  def __init__(self, properties):
    super().__init__(properties)

  def crawl(self):
    lookml_maker = LookMaker("")
    lookml_maker.write_all_models_to_vdb()

class SqlConverter(metaclass=ABCMeta):
  def __init__(self, question):
    self.callbacks = []
    self.converted_sql = ""
    self.filters = []
    self.question_history = []
    self.question_history.append(question)
    self.last_event = SqlConverterEventType.INIT
    self.result = None
    self.vdb = VectorDatabase()
  
  def register_callback(self, callback : Callable[[SqlConverterEventType, str], any]):
    if callback not in self.callbacks:
      print("call back registered: " + str(callback))
      self.callbacks.append(callback)
    else:
      print("call back not registered: " + str(callback))

  def invoke_callback(self, event_type : SqlConverterEventType, message : str):
    self.last_event = event_type
    for callback in self.callbacks:
      callback(event_type, message)
  
  def set_result(self, result_type, converted_sql, result_set, dashboard_url, look_url):
    self.result = SqlConverterResult(result_type, converted_sql, result_set, dashboard_url, look_url)

  def get_result(self):
    return self.result
  
  def suggest_additional_information(self, message : str):
    self.question_history.append(message)

  def get_field_unique_values(self, matched_table, matched_field):
    if matched_table[0] != '`' :
      matched_table = '`' + matched_table + '`'
    sql_query = f"with distinct_values as ( select distinct {matched_field} as {matched_field} from {matched_table} ) select {matched_field}, (select count(1) from distinct_values) as total_count from distinct_values limit 500"
    df = client.query(sql_query).to_dataframe()
    return df[matched_field].tolist(), df['total_count'][0]

  @abstractmethod
  def try_convert(self):
    pass
  
  def convert(self):
    self.invoke_callback(SqlConverterEventType.PROGRESSING, "Converting SQL...")
    try:
      event_type, message = self.try_convert()
      if event_type == SqlConverterEventType.NEED_MORE_INFO:
        self.invoke_callback(SqlConverterEventType.NEED_MORE_INFO, message)
        return
      self.invoke_callback(SqlConverterEventType.FINISHED, "Finished")
    except Exception as e:
      print(e)
      self.invoke_callback(SqlConverterEventType.ERROR, e)
  


class DirectSqlConverter(SqlConverter):

  def __init__(self, question):
    super().__init__(question=question)

  def try_convert(self):
    self.invoke_callback(SqlConverterEventType.PROGRESSING, "Finding related tables...")
    self.related_tables = self.get_related_tables()
    self.invoke_callback(SqlConverterEventType.PROGRESSING, "Generating SQL with schema...")
    self.converted_sql = self.convert_sql_with_schemas()
    self.invoke_callback(SqlConverterEventType.PROGRESSING, "Extracing filter values...")
    self.sql_and_filters = self.extract_filter_columns()
    self.invoke_callback(SqlConverterEventType.PROGRESSING, "Adjusting filter values...")
    self.adjust_filter_value(self.sql_and_filters['filter_columns'])
    self.invoke_callback(SqlConverterEventType.PROGRESSING, "Replacing filter values...")
    self.choose_right_one_value_from_adjusted_values()
    self.invoke_callback(SqlConverterEventType.PROGRESSING, "Executing the query...")
    df_result = self.prepared_statement_with_filter_values_in_bigquery()
    self.set_result(SqlConverterResultType.RESULT_SET, self.converted_sql, df_result, None, None)
    return SqlConverterEventType.FINISHED, "Sucess"
  
  def get_formatted_schema(self, table_name, table_description, column_schema):
    #column_schema_template = """      {table_name}.{column_name} {column_type} # {column_description}"""
    column_schema_template = """      {column_name} {column_type} """
    table_schema_template = """  * table name : {table_name} REMARKS '{table_description}'
    * columns :
    (
{column_schema}
    )

"""
    column_schema_list = []
    for column in parse_python_object(column_schema):
      #column_schema_list.append(column_schema_template.format(table_name=table_name,column_name=column['column_name'], column_description=column['column_description'], column_type=column['column_type']))
      column_schema_list.append(column_schema_template.format(column_name=column['column_name'], column_type=column['column_type']))
    column_schema_str = "\n".join(column_schema_list)
    return table_schema_template.format(table_name=table_name, table_description=table_description, column_schema=column_schema_str)

  def get_related_tables(self):
    test_embedding =  embeddings.embed_query(str(self.question_history))
    results = []
    with self.vdb.get_connection() as conn:
      try:
        with conn.cursor() as cur:
          select_record = (str(test_embedding).replace(' ',''),)
          cur.execute(f"SELECT table_name, description, column_schema FROM rag_test where (1 - (desc_vector <=> %s)) > 0.5 ", select_record)
          results = cur.fetchall()
          #print(results)
      except Exception as e:
        print(e)
        raise e
    return results

  def convert_sql_with_schemas(self):
    prompt_template = """You are a Developer, convert the following question into SQL with the given schema information:

related_schemas :
{related_tables}

question :
{question}

output: SQL
"""
    related_table_list = []
    for related_table in self.related_tables:
      related_table_list.append(self.get_formatted_schema(related_table[0], related_table[1], related_table[2]))
    related_table_str = "\n".join(related_table_list)
    #print(related_table_str)
    prompt = prompt_template.format(related_tables=related_table_str, question=str(self.question_history))
    #print(prompt)
    response = llm.predict(prompt)
    #print(response)
    return response


  def extract_filter_columns(self):
    sample_json = """
    {
      "prepared_statement" : "select * from `bigquery-public-data.thelook_ecommerce.delivery` where created_at between ? and ?",
      "filter_columns" : [
        {
          "table_name" : "bigquery-public-data.thelook_ecommerce.delivery",
          "column_name" : "created_at",
          "column_type" : "TIMESTAMP",
          "operator" : "between",
          "filter_names" : ["created_at_start", "created_at_end"],
          "filter_values" : ["2020-01-01", "2020-01-02"],
          "filter_order" : 1
        }
      ]
    }
    """
    prompt_template = """You are a looker developer, extract the filter columns and change the given sql into prepared statement in JSON format. Please don't suggest python code. Give me a json output as the given output example format.:

    output format : json
    {sample_json}

    ----------------------------------------------
    sql :
    {sql}

    related_tables :
    {related_tables}

    """
    prompt = prompt_template.format(sql=self.converted_sql, parameter_char=PREPARED_STATEMENT_PARAMETER_CHAR, related_tables=self.related_tables, sample_json=sample_json)
    response = llm.predict(prompt)
    return parse_json_response(response)

  def choose_right_filter_value(self, filter_values, wanted_value):
    prompt_template = """As a looker developer, choose right filter value for the wanted value below without changing filter value itself.

    filter_values : {filter_values}

    wanted_values: {wanted_value}

    answer format: python list
  [filter_value1, filter_value2, ...]
    """
    prompt = prompt_template.format(filter_values=filter_values, wanted_value=wanted_value)
    response = llm.predict(prompt)
    return response 

  def adjust_filter_value(self, filter_columns):
    for filter in filter_columns:
      matched_table = filter['table_name']
      matched_field = filter['column_name']
      filter['unique_values'], filter['unique_count'] = self.get_field_unique_values(matched_table, matched_field)
      # TODO: if unique_count < 500, then choose right filter value in the unique value list.
      if filter['unique_count'] < 500:
        response = self.choose_right_filter_value(filter['unique_values'], filter['filter_values'])
        print(response)
        if response.strip().find("```json") == 0 :
          filter['adjust_filter_values'] = parse_json_response(response)
        else:
          filter['adjust_filter_values'] = parse_python_object(response)
      else:
        filter['adjust_filter_values'] = filter['filter_values']
  
  SINGLE_OPERATORS = ['=', '>', '<', '>=', '<=', '!=', '<>']

  def choose_right_one_value_from_adjusted_values(self):
    for filter in self.sql_and_filters['filter_columns']:
      if filter['operator'] in DirectSqlConverter.SINGLE_OPERATORS :
        filter['adjust_filter_values'] = [filter['adjust_filter_values'][0]]

  def prepared_statement_with_filter_values_in_bigquery(self):
    prepared_statement = self.sql_and_filters['prepared_statement']
    query_parameters = []
    for filter_column in self.sql_and_filters['filter_columns']:
      if len(filter_column['adjust_filter_values']) > 1:
        if(filter_column['column_type'] == 'FLOAT64'):
          query_parameters.append(bigquery.ArrayQueryParameter(None, "FLOAT64", filter_column['adjust_filter_values']))
        elif(filter_column['column_type'] == 'INT64'):
          query_parameters.append(bigquery.ArrayQueryParameter(None, "INT64", filter_column['adjust_filter_values']))
        else:
          query_parameters.append(bigquery.ArrayQueryParameter(None, "STRING", filter_column['adjust_filter_values']))  
      else:
        if(filter_column['column_type'] == 'FLOAT64'):
          query_parameters.append(bigquery.ScalarQueryParameter(None, "FLOAT64", filter_column['adjust_filter_values'][0]))
        elif(filter_column['column_type'] == 'INT64'):
          query_parameters.append(bigquery.ScalarQueryParameter(None, "INT64", filter_column['adjust_filter_values'][0]))
        else:
          query_parameters.append(bigquery.ScalarQueryParameter(None, "STRING", filter_column['adjust_filter_values'][0]))
    job_config = bigquery.QueryJobConfig(
      query_parameters=query_parameters
    )
    print(prepared_statement)
    query_job = client.query(prepared_statement, job_config=job_config)
    return query_job.to_dataframe()


def dummy_callback(event_type : SqlConverterEventType, message : str):
  print(event_type)
  print(message)

if __name__ == "__main__":

  # crawler = BigQuerySchemaScrawler({'dataset_id' : 'bigquery-public-data.thelook_ecommerce'})
  # crawler.truncate_table()
  # crawler.crawl()
  # carwler test finished

  # sql_converter = DirectSqlConverter("I want to know the total count of the product in Sports category.")
  # sql_converter.register_callback(dummy_callback)
  # sql_converter.convert()
  # print(sql_converter.get_result().get_converted_sql())
  # print(sql_converter.get_result().get_result_set())
  # print(sql_converter.get_result().get_dashboard_url())
  # print(sql_converter.get_result().get_look_url())

  # crawler = LookerNavExplorerCrawer({})
  # crawler.crawl()
  pass
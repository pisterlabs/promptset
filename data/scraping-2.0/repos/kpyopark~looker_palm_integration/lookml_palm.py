import looker_sdk
import vertexai
import os
import json
from langchain.chat_models import ChatVertexAI
from langchain.llms import VertexAI
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field, validator
from looker_sdk.sdk.api31 import models as ml
from typing import cast, Dict, List, Union
from vector_util import VectorDatabase
from langchain.embeddings import VertexAIEmbeddings

class LookerFilterValue(BaseModel):
  field_name: str = Field(description="field_name")
  values: List[str] = Field(description="values")

class LookerSortField(BaseModel):
  field_name: str = Field(description="field_name")
  direction: str = Field(description="direction")

class LookerQueryParameters(BaseModel):
  dimensions: List[str] = Field(description="dimensions")
  measures: List[str] = Field(description="measures")
  parameters: List[str] = Field(description="parameters")
  filters: List[LookerFilterValue] = Field(description="filters")
  sorts: List[LookerSortField] = Field(description="sorts")
  pivots: List[str] = Field(description="pivot - These fields are used as pivots in the chart.")
  hidden_fields: List[str] = Field(description="hidden_fields - These fields are used as filters but are not shown in the chart.")

class LookerFilterRetrieves(BaseModel):
  required_target: List[str] = Field(description="required_target")

class LookMaker():

  def __init__(self, question):
    self.question = question
    self.llm = None
    self.lookml_explore = None
    self.lookml_model = None
    self.fields = None
    self.schema = None
    self.related_fields = None
    self.valid_filter_values = None
    self.filter_dict:Dict[str, str] = {}
    self.PROJECT_ID = os.getenv("PROJECT_ID")  # @param {type:"string"}
    self.location = "us-central1"
    self.is_public_publishing = True
    self.vdb = VectorDatabase()
    self.init_llm()
    self.init_sdk()

  def init_llm(self):
    vertexai.init(project=self.PROJECT_ID, location=self.location)
    self.llm = VertexAI(
      model_name="text-bison-32k",
      max_output_tokens=8000,
      temperature=0,
      top_p=0.8,
      top_k=40,
    )
    self.embeddings = VertexAIEmbeddings()
  
  def init_sdk(self):
    # instantiate sdk
    #self.sdk = looker_sdk.init31()
    self.sdk = looker_sdk.init40()

  def write_navexplore_to_vdb(self, lookml_model):
    for nav_explore in lookml_model.explores:
      if(nav_explore.description is None):
        continue 
      description = nav_explore.description
      desc_vector = self.embeddings.embed_query(description)
      self.vdb.insert_record(sql=None, parameters=None, description=description, explore_view=nav_explore.name, model_name=lookml_model.name, table_name=None, column_schema=None, desc_vector=desc_vector)

  def write_all_models_to_vdb(self):
    for lookml_model in self.sdk.all_lookml_models():
      self.write_navexplore_to_vdb(lookml_model)

  def choose_right_explore(self) -> None:
    # self.lookml_explore = "national_pension_mom"
    # self.lookml_model = "lookml_hol_sample"
    test_embedding =  self.embeddings.embed_query(self.question)
    with self.vdb.get_connection() as conn:
      try:
        with conn.cursor() as cur:
          select_record = (str(test_embedding).replace(' ',''),)
          cur.execute(f"with tb_settings as (select %s::vector as compared_vec) SELECT model_name, explore_view, description, 1 - cosine_distance(desc_vector,compared_vec) as cosine_sim FROM rag_test, tb_settings where (1 - cosine_distance(desc_vector,compared_vec)) > 0.5 order by 4 desc limit 4", select_record)
          result = cur.fetchone()
          print(result[0], result[1], result[2], result[3])
          self.lookml_model = result[0]
          self.lookml_explore = result[1]
      except Exception as e:
        print(e)
    return None, None

  def get_schema_for_the_explore(self) -> None:
    print(self.lookml_model + ":" + self.lookml_explore)
    # API Call to pull in metadata about fields in a particular explore
    explore = self.sdk.lookml_model_explore(
      lookml_model_name=self.lookml_model,
      explore_name=self.lookml_explore,
      fields="id, name, description, fields",
    )

    my_fields = []

    # Iterate through the field definitions and pull in the description, sql,
    # and other looker tags you might want to include in  your data dictionary.
    if explore.fields and explore.fields.dimensions:
      for dimension in explore.fields.dimensions:
        dim_def = {
          "field_type": "Dimension",
          "view_name": dimension.view_label,
          "field_name": dimension.name,
          "type": dimension.type,
          "description": dimension.description,
          #"sql": dimension.sql,
        }
        my_fields.append(dim_def)
    if explore.fields and explore.fields.measures:
      for measure in explore.fields.measures:
        mes_def = {
          "field_type": "Measure",
          "view_name": measure.view_label,
          "field_name": measure.name,
          "type": measure.type,
          "description": measure.description,
          #"sql": measure.sql,
        }
        my_fields.append(mes_def)
    if explore.fields and explore.fields.parameters:
      for parameter in explore.fields.parameters:
        par_def = {
          "field_type": "Parameter",
          "view_name": parameter.view_label,
          "field_name": parameter.name,
          "default_filter_value": parameter.default_filter_value,
          "type": parameter.type,
          "description": parameter.description,
          #"sql": parameter.sql,
        }
        my_fields.append(par_def)
    self.schema = my_fields

  def get_field_type(self, field_name) -> str:
    for field in self.schema:
      if field['field_name'] == field_name:
        return field['type']

  def parse_llm_reponse_to_fields_object(self, response) -> LookerQueryParameters:
    parser = PydanticOutputParser(pydantic_object=LookerQueryParameters)
    return parser.parse(response)

  def choose_related_fields(self) -> None:
    sample_json = """
    {
      "dimensions": [
        "dimension1",
      ],
      "measures": [
        "measure1",
      ],
      "filters": [
        {
          "field_name": "field_name1",
          "values": [
            "value1"
          ]
        }
      ],
      "sorts": [
        {
          "field_name": "field_name1",
          "direction": "asc"
        }
      ],
      "parameters": [
        "param1",
      ],
      "pivots": [
        "field1"
      ],
      "hidden_fields": [
        "field1"
      ]
    }
    """

    prompt_template = """As a looker developer, choose right dimesions and measures for the question below. 
    You should choose right fields as least as possible and sort fields must be choosen in the dimension fields.

    fields : {fields}

    question: {question}

    answer format: json
    {sample_json}
    """
    response = self.llm.predict(prompt_template.format(fields=self.schema, question=self.question, sample_json=sample_json))
    self.related_fields = self.parse_llm_reponse_to_fields_object(response)

  def parse_llm_response_to_retreive_target_filters(self, retrieve_target_filters) -> LookerFilterRetrieves:
    parser = PydanticOutputParser(pydantic_object=LookerFilterRetrieves)
    return parser.parse(retrieve_target_filters)

  def get_user_input_value_for_filter_field(self, field_name) -> str:
    for filter in self.related_fields.filters:
      if filter.field_name == field_name:
        return filter.values
    return ""

  def decide_to_retrieve_values_for_the_filters(self) -> None:
    # output_sample = """
    # {
    #   "required_target": ["field1","field2"]
    # }
    # """
    # prompt_template = """As a looker developer, decide whether to retrieve values for the filters below. 
    # For example, date / timestamp columns don't need to retrieve values. but string columns need to retrieve values from the database.

    # filter fields : {filter_fields}

    # output sample : json array
    # {output_sample}
    # """
    #response = self.llm.predict(prompt_template.format(filter_fields=self.related_fields.filters, output_sample=output_sample))
    #self.retrieve_target_filters = self.parse_llm_response_to_retreive_target_filters(response)
    required_target = []
    for filter in self.related_fields.filters:
      field_type = self.get_field_type(filter.field_name)
      # if field_type == 'string':
      required_target.append(filter.field_name)
    self.retrieve_target_filters = LookerFilterRetrieves(required_target=required_target)

  # def get_value_list_from_json_array(self, json_array):
  #   values = []
  #   for json_object in json_array:
  #     print(json_object)
  #     values.append(list(json_object.values())[0])
  #   return values

  def get_validated_filter_values_from_looker(self) -> None:
    choose_right_filter_value_list = []
    for retrieve_target_filter in self.retrieve_target_filters.required_target:
      #print(retrieve_target_filter)
      query_template = ml.WriteQuery(model=self.lookml_model, view=self.lookml_explore,fields=[retrieve_target_filter])
      query = self.sdk.create_query(query_template)
      # json_object = json.loads(self.sdk.run_query(query.id, "json"))
      # choose_right_filter_value_list.append({ retrieve_target_filter : self.get_value_list_from_json_array(json_object)})
      choose_right_filter_value_list.append({ retrieve_target_filter : self.sdk.run_query(query.id, "json")})
    self.retrieve_filter_and_values = choose_right_filter_value_list

  def parse_json_response(self, llm_json_response) -> any:
    parsed_json = None
    try :
      start_char = '['
      end_char = ']'
      if llm_json_response.find('[') == -1 or llm_json_response.find('{') < llm_json_response.find('[') :
        start_char = '{'
        end_char = '}'
      start_index = llm_json_response.find(start_char)
      end_index = llm_json_response.rfind(end_char)
      json_data = llm_json_response[start_index:end_index+1]
      json_data = json_data.replace('\\n', '')
      parsed_json = json.loads(json_data)
    except Exception as ex:
      print(ex)
      print("json parse error: " + json_data)
    return parsed_json

  def choose_right_filter_value(self, filter_values, wanted_value) -> any:
    example_json = "[{\"national_pension_mom.data_create_yearmonth_year\":2022}]"
    prompt_template = """As a looker developer, choose right filter value for the wanted value below without changing filter value itself.

    example :
    {example_json}

    filter_values : {filter_values}

    wanted_values: {wanted_value}

    answer format: json array
    """
    response = self.llm.predict(prompt_template.format(example_json=example_json,filter_values=filter_values, wanted_value=wanted_value))
    print("Choose Right Filter Value:" + response)
    return self.parse_json_response(response)

  def get_appropriate_filter_value_pair(self) -> None:
    self.valid_filter_values = []
    for filter_and_values in self.retrieve_filter_and_values:
      field_name = list(filter_and_values.keys())[0]
      user_input_value = self.get_user_input_value_for_filter_field(field_name)
      value_object = self.choose_right_filter_value(filter_and_values, user_input_value)
      self.valid_filter_values.append(value_object)
      filter_and_values[field_name] = value_object

  def get_quoted_value(self, field_name) -> str:
    values = []
    for filter_values in self.valid_filter_values:
      print(filter_values)
      for filter_value in filter_values:
        field_name_cmp = list(filter_value.keys())[0]
        field_value = list(filter_value.values())[0]
        field_type = self.get_field_type(field_name)
        if field_name_cmp == field_name:
          if field_type == 'string':
            values.append(field_value)
          else:
            values.append(str(field_value))
    return ','.join(values)
  
  def get_lookml_filter_dictionary(self) -> None:
    self.filter_dict:Dict[str, str] = {}
    for filter in self.related_fields.filters:
      field_name = filter.field_name
      quoted_values = self.get_quoted_value(field_name)
      filter.values = quoted_values
      self.filter_dict[field_name] = quoted_values

  def make_dimension_and_description_pair(self) -> List[str]:
    dimension_and_description_pair = []
    for one_dimension in self.related_fields.dimensions:
      for dimension in self.schema:
        if dimension['field_name'] == one_dimension:
          dimension_and_description_pair.append((one_dimension, dimension['description']))
    return dimension_and_description_pair

  def choose_chart_type_and_pivots(self) -> any:
    dimension_and_description_pair = self.make_dimension_and_description_pair()
    sample_json = """{
    "chart_type": "looker_column",
    "date_time_dimensions": ["dimension1"],
    "pivots": [
      "field1"
    ],
    "hidden_fields": [
      "field1"
    ]
    "reason_to_choose": "I choose field1 as a pivot field because ..."
    }"""
    prompt_template = """As a looker developer, choose chart type and pivot fields and hidden fields in the given dimensions for the question below. 
    Pivot field is a field that is used to create a pivot table. A pivot field converts category values in the field to columns so that you can compare different category values. 
    For example, if you have sales data, you can compare sales by product by setting the "Product" field as a pivot field. Date/time fields MUST not be used as pivot fields.
    Hidden field is a field that is not displayed in a chart. Hidden fields are used to hide fields that are not needed in the chart or that can confuse users. 
    For example, the "Product ID" field can be used to identify products, but it does not need to be displayed in a chart. If there are two same date fields, one of them can be hidden. 
    At least one dimension field must be a visible field that is not used in pivot fields or hidden fields.

    chart_types : 
    looker_column - Column charts are useful when you want to compare the values of multiple fields(under 3~4 categories) for multiple records. It needs one main field to show the values separated by the main field. And this field must not be a pivot field.
    looker_line - Line charts are useful when you want to show the changes in a value over time. They are also useful for comparing the changes in many categories over time.
    looker_area - Area charts are useful when you want to compare the trends of two or more values over time. They are also useful for showing the cumulative sum of values over time.
    looker_funnel - Funnel charts are useful to understand events in a sequential process, like prospect stages in a sales pipeline, engagement with a marketing campaign, or visitor movement through a website.
    looker_pie - Pie charts are useful when you want to show the proportion of values to the total value. They are also useful for comparing the proportional differences between values. Pivot fields are not allowed.
    looker_timeline - Timeline charts are useful when you want to show events over time. They are also useful for showing the duration of events. It needs at least 3 fields. 1. Event Name 2. Start Date 3. End Date
    looker_table - Table charts are useful when you want to show the values of multiple fields for multiple records. They are also useful for showing the values of multiple fields for a single record.

    dimensions : 
    {dimensions}

    question:
    {question}

    answer format: json
    {sample_json}
    """
    prompt_full = prompt_template.format(dimensions=dimension_and_description_pair, question=self.question, sample_json=sample_json)
    response = self.llm.predict(prompt_full)
    return self.parse_json_response(response)
  
  def make_query(self):
    self.choose_right_explore()
    self.get_schema_for_the_explore()
    self.choose_related_fields()
    self.decide_to_retrieve_values_for_the_filters()
    self.get_validated_filter_values_from_looker()
    self.get_appropriate_filter_value_pair()
    self.get_lookml_filter_dictionary()

    fields = []
    fields.extend(self.related_fields.dimensions)
    fields.extend(self.related_fields.measures)
    filters = self.filter_dict
    chart_type_and_pivots = self.choose_chart_type_and_pivots()
    hidden_fields = chart_type_and_pivots['hidden_fields']
    pivot_fields = chart_type_and_pivots['pivots']
    chart_type = chart_type_and_pivots['chart_type']
    print('fields:' + str(fields))
    print('filters:' + str(filters))
    print('hidden_fields:' + str(hidden_fields))
    print('pivot_fields:' + str(pivot_fields))
    print('chart_type:' + str(chart_type))
    query_template = ml.WriteQuery(model=self.lookml_model, view=self.lookml_explore, fields=fields, filters=filters, pivots=pivot_fields, query_timezone="Asia/Seoul", vis_config={'type':chart_type, 'hidden_fields':hidden_fields})
    query = self.sdk.create_query(query_template)
    run_response = self.sdk.run_inline_query("json", query)
    print('query.id:' + str(query.id))
    self.query = query

  def make_look(self):
    self.make_query()
    generated_query = self.query
    existing_look = self.sdk.search_looks(query_id=generated_query.id)
    if len(existing_look) > 0:
      return existing_look[0]
    existing_look = self.sdk.search_looks(title=self.question)
    if len(existing_look) > 0:
      return existing_look[0] 
    new_look = self.sdk.create_look(ml.WriteLookWithQuery(query_id=generated_query.id, 
      description=self.question,
      deleted=False,
      is_run_on_load=True,
      public=self.is_public_publishing,
      folder_id=str(self.sdk.me().personal_folder_id),
      title=self.question+"4"))
    return new_look


if __name__ == "__main__":
  maker = LookMaker("22년 5월부터 6월까지 보건업 연금 납부 평균액을 월별 그래프로 보여줘")

  look = maker.make_look()
  print(look.id)
  print(look.short_url)
  print(look.public_url)

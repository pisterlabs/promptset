from datetime import datetime
from typing import ForwardRef, Union, get_args, get_origin

from langchain import SQLDatabase, SQLDatabaseChain
from langchain.base_language import BaseLanguageModel
from langchain.prompts.prompt import PromptTemplate
from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Integer, MetaData, String, Table, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import CreateTable

from ...client.types import Unset


class SQLParser:
    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm

    def parse(self, input, query):
        # Check if the input is a list (the SQL parser is only available for resources that return lists)
        if not (isinstance(input, list) or get_origin(input) is list):
            raise TypeError("SQL parser is only available for resources that return lists")

        input_class = type(input[0])
        input_class_name = input_class.__name__

        class_attrs = {
            "__tablename__": input_class_name.lower(),
            "temporary_id": Column(Integer, primary_key=True),
        }

        # Could maybe use SQLAlchemy or another library like SQLAlchemy Utils for this mapping
        def map_sql_types(type_var):
            if type_var is str:
                return String
            elif type_var is int:
                return Integer
            elif type_var is datetime:
                return DateTime
            elif type_var is float:
                return Float
            elif type_var is dict:
                return JSON
            elif type_var is bool:
                return Boolean
            elif get_origin(type_var) is dict:
                return JSON
            elif get_origin(type_var) is list:
                return JSON
            elif type(type_var) == ForwardRef:
                return JSON
            else:
                raise ValueError(f"Unknown type when mapping Typing to SQLAlchemy: {type_var}")

        # TODO check if there's a better way to pull these
        for field_name, field_type in input_class.__dict__["__annotations__"].items():
            if get_origin(field_type) is Union:
                # First argument should be Unset, otherwise raise an error
                if get_args(field_type)[0] is not Unset:
                    raise ValueError("Union type with first value not Unset not supported")
                class_attrs[field_name] = Column(map_sql_types(get_args(field_type)[1]), nullable=True)
            else:
                class_attrs[field_name] = Column(map_sql_types(field_type))

        engine = create_engine(f"sqlite:///{input_class_name.lower()}.db", echo=True)
        Base = declarative_base()
        sql_class = type(input_class_name, (Base,), class_attrs)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()

        data = []
        for line in input:
            data.append(line.to_dict())
        session.bulk_insert_mappings(sql_class, data)
        session.commit()
        session.close()

        metadata = MetaData()
        metadata.reflect(bind=engine)
        sql_table = Table(input_class_name.lower(), metadata, autoload=True, autoload_with=engine)

        template_table = "\nOnly use the following tables:\n" + str(CreateTable(sql_table).compile(engine))
        # If the table isn't too large, would be better to also include example data
        template_start = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer. Use the following format:

        Question: "Question here"
        SQLQuery: "SQL Query to run"
        SQLResult: "Result of the SQLQuery"
        Answer: "Final answer here"
        """
        template_end = "Question: {input}"

        prompt_template = template_start + template_table + template_end

        prompt = PromptTemplate(input_variables=["input", "dialect"], template=prompt_template)
        db = SQLDatabase.from_uri(f"sqlite:///{input_class_name.lower()}.db")
        db_chain = SQLDatabaseChain.from_llm(llm=self.llm, db=db, prompt=prompt, verbose=True)
        res = db_chain.run(query)

        # Drop the created table so it doesn't overwrite later
        sql_table.drop(engine)

        return res

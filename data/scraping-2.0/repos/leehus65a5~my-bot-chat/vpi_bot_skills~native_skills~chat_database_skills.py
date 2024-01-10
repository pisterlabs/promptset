from semantic_kernel.skill_definition import sk_function, sk_function_context_parameter
from semantic_kernel import SKContext
from semantic_kernel.sk_pydantic import PydanticField

import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
import os
from sqlalchemy import create_engine
from sqlalchemy.sql import text

class ChatWithDataBase(PydanticField):
 
	@sk_function(
	name= "execute",
	description="execute the sql query and return resutl",
	input_description="user query"
	)
	async def get_table_name(self, context: SKContext) -> str:
		connector = create_engine('mysql://root:''@localhost/student').connect()
		sql_query = text(context['query'])
		try:
			result = connector.execute(sql_query).fetchall()
			return(str(result))
		except:
			return None

	


from ..dataloaders.DeepLakeLoader import DeepLakeLoader
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from api.web.prompts import DETECT_OBJECTION_PROMPT, OBJECTION_GUIDELINES_PROMPT
from sqlalchemy.ext.asyncio import AsyncSession
from ..dto import request_model
from fastapi import HTTPException
from ..repositories import repositories
from ..dao import db_model

async def initialize_deeplake(query: request_model.CreateUserData):
    db = DeepLakeLoader('data/SalesPilot.txt', token_id=query.activeloop_token_id)
    return db

async def get_response(user_data: request_model.CreateUserData, query_data: request_model.CreateQuery, session: AsyncSession):
    deeplake_obj = await initialize_deeplake(user_data)
    user_query = query_data.query
    chat = ChatOpenAI(openai_api_key=user_data.chatgpt_api_key)
    system_message = SystemMessage(content = DETECT_OBJECTION_PROMPT)
    human_message = HumanMessage(content = f"{user_query}")
    response = chat([system_message, human_message])
    if response.content == "None":
        detected_objection = "No sales objection found in query"
    else:
        detected_objection = response.content
    results = deeplake_obj.query_db(detected_objection)
    if results:
        system_message = SystemMessage(content=OBJECTION_GUIDELINES_PROMPT)
        human_message = HumanMessage(content=f'Customer objection: {detected_objection} | Relevant guidelines: {results}')

        response = chat([system_message, human_message])
        print("type", type(response), "response:", response)
    else:
        response = AIMessage(content="No recommendation provided for the sales objection")
    #create query and response entry in Query table
    query_entry = request_model.CreateQuery(
        query=user_query,
        sales_objection=detected_objection,
        response=response.content
    )
    created_query = await repositories.create_query(user_query = query_entry, session = session)

    return created_query

async def get_all_queries(session: AsyncSession):
    queries = await repositories.get_all_queries(session = session)
    if queries:
        return queries
    else:
        raise HTTPException(
            status_code=400,
            detail="No queries found!"
        )
    
async def delete_query_by_id(id: int, session: AsyncSession):
    query_data = await repositories.get_data_by_column(
        column = db_model.Query.id,
        value= id, 
        session= session)
    if query_data:
        #delete from Urls table
        deleted_query = await repositories.delete_query(
            session = session, query_obj = query_data
        )
        return deleted_query
    else:
        raise HTTPException(
            status_code=400,
            detail = "Query id does not exists!"
        )



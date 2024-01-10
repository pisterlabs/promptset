from pydantic import BaseModel, Field
from pprint import pprint
from typing import Annotated, Type, Any, List

# APP related
from app.api.api_v1.models import (
    UserMessage,
    BotMessage,
    WebsiteQuestion,
    AddSource,
    Datasource,
)
from app.api.api_v1.utils import search_pinecone, load_file
from app.api.api_v1.database import (
    retrieve_chat_history,
    insert_chat_history,
    upsert_datasource,
    retrieve_agent_datasources,
)

# LLMs and LangChain
from langchain.chat_models.openai import ChatOpenAI
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import Tool
from langchain.tools import StructuredTool
from langchain.chains import LLMMathChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import AgentExecutor

# Data analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_experimental.agents.agent_toolkits import (
    create_pandas_dataframe_agent,
)


"""" AGENT TOOLS """

ai_agent_name = "Information Systems"


class MetadataFilters(BaseModel):
    """Filters used by the vectorstore (Pinecone) to search on metadata."""

    file_link: List[str] | None = Field(
        description="Source file_link that the data came.  This field can be used to match against when comparing to the 'file_link' property in the Datasource (returned from the list_sources_tool)",
        default=None,
    )
    before_indexed_datetime_utc: str | None = Field(
        description="The datetime in UTC format when the data was added to the VectorStore. This field can be used to only return data that was added before a specific time period. Use a standard mongo style query to construct the condition",
        default=None,
    )
    after_indexed_datetime_utc: str | None = Field(
        description="The datetime in UTC format when the data was added to the VectorStore. This field can be used to only return data that was added after a specific time period. Use a standard mongo style query to construct the condition",
        default=None,
    )
    original_file_type: List[str] | None = Field(
        description="The original file type(s). ", default=None
    )
    filter_operator: str | None = Field(
        description="The mongo style logical operator used to combine multiple filters. Use $or to match any filter or $and to match all filters.",
        default="$or",
    )


class RagArgs(MetadataFilters):
    query: str = Field(
        description="The text content, or meaning content, to search for in the VectorStore of previously saved datasources."
    )
    texts_only: bool | None = Field(
        description="Use False to get the full VectorStore response in JSON which includes the metadata Score (The highest is the most likely answer so favor the text from that resul).  Use True to only get the texts from the VectorStore",
        default=False,
    )


async def construct_metadata_filter(
    mdf: MetadataFilters, operator: str = "$or"
) -> dict:
    filters = []
    filter = {}
    for key, value in mdf.model_dump().items():
        pprint(f"{key}:{value}")
        match str(key):
            case "file_link":
                if value is not None:
                    filters.append({"source": {"$in": [s for s in value]}})
            case "before_indexed_datetime_utc":
                if value is not None:
                    filters.append({"indexed_datetime_utc": {"$gt": value}})
            case "after_indexed_datetime_utc":
                if value is not None:
                    filters.append({"indexed_datetime_utc": {"$lt": value}})
            case "original_file_type":
                if value is not None:
                    filters.append({"original_file_type": {"$in": [s for s in value]}})

    if len(filters) > 1:
        filter = {operator: [f for f in filters]}
    elif len(filters) == 1:
        filter = filters[0]
    return filter


async def rag(
    query: str,
    file_link: List[str] | None = None,
    before_indexed_datetime_utc: str | None = None,
    after_indexed_datetime_utc: str | None = None,
    original_file_type: List[str] | None = None,
    texts_only: bool = False,
    filter_operator: str | None = "$or",
) -> str | None:
    """Useful for finding information in a previously saved source. Returns either a list of JSON objects or just texts depending on usage of the texts_only argument."""

    mf = {}
    m = MetadataFilters(
        file_link=file_link,
        before_indexed_datetime_utc=before_indexed_datetime_utc,
        after_indexed_datetime_utc=after_indexed_datetime_utc,
        original_file_type=original_file_type,
    )
    mf = await construct_metadata_filter(mdf=m, operator=filter_operator)

    context = await search_pinecone(
        query=query, top_k=3, texts_only=texts_only, metadata_filter=mf
    )

    return context


def pandas_agent(csv_path: str, question: str) -> Any:
    """Useful for doing anyalysis from a CSV.  Use the question parameter to specify the question for pandas agent"""

    df = pd.read_csv(csv_path)

    llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.0)
    # Show the intermediate steps because this agent sometimes doesn't put the full answer in the final answer.
    agent = create_pandas_dataframe_agent(
        llm, df=df, verbose=True, return_intermediate_steps=True
    )

    return agent(question)


async def add_source(
    ai_agent: str,
    file_type: str,
    file_link: str,
    created_by: str,
    title: str | None = None,
    sparse_summary: str | None = None,
    recursive_scraping: bool | None = True,
) -> str | bool:
    """Useful when a user wishes to add a data source. Make sure to create a title and a sparse summary for the user if they don't provide one. If the link goes to an html page or website, you should ask the uer if the want to do 'Recursive Scraping' or just scrape that one page"""
    # metadata for the vectorstore
    meta = []
    ai_agent = ai_agent_name
    if title is not None:
        meta.append({"title": title})
    # load the file in the vectorstore and backup to s3
    s3_key = await load_file(
        file_type=file_type,
        file_link=file_link,
        metadata_to_save=meta,
        recursive_scraping=recursive_scraping,
    )
    if s3_key:
        # file loaded, now add to the datasource collection in mongo
        ds = Datasource(
            file_link=file_link,
            file_type=file_type,
            ai_agent=ai_agent,
            created_by=created_by,
            title=title,
            sparse_summary=sparse_summary,
            s3_key=s3_key,
        )
        new_ds = await upsert_datasource(ds)

    return s3_key


async def list_sources() -> list[Datasource] | None:
    """Useful for getting a list of all the currently saved data sources that the you have access to. Use the rag tool to get information within these sources"""
    print("RUNNING LIST SOURCES")
    r = await retrieve_agent_datasources(agent=ai_agent_name)
    pprint(r)
    return r


""" AGENT """


async def agent(payload: UserMessage):
    llm = ChatOpenAI(model="gpt-4-1106-preview")

    rag_tool = StructuredTool.from_function(rag)
    rag_tool.coroutine = rag
    rag_tool.args_schema = RagArgs

    add_source_tool = StructuredTool.from_function(add_source)
    add_source_tool.coroutine = add_source
    add_source_tool.args_schema = Datasource

    list_sources_tool = StructuredTool.from_function(list_sources)
    list_sources_tool.coroutine = list_sources

    pandas_agent_tool = StructuredTool.from_function(pandas_agent)
    # pandas_agent_tool.return_direct = True

    tools = [rag_tool, add_source_tool, list_sources_tool, pandas_agent_tool]

    system_prompt = f"""
        You are an ambitious and friendly genius named Salkie, internally your ai_agent name is {ai_agent_name}. You have degrees in Information Systems, Business Administration, Logic, Liberal Arts, Law and Computer Science. You have the amazing ability to read JSON data and make sense of it easily.  You are presently working for a high ranking employee at the Salk Institute who's username is {payload.username}, you help them answer and plan their next action for any questions, challenges or research they are doing.
        If using the rag tool always use the ENTIRE question from the user for the query parameter.
        For any tool you want to use, make sure you have values for all of the tool's requied parameters, otherwise don't use that tool.
        You love to respond in Markdown syntax and always with uniquely cited sources (as clickable links) at the end. You will use as many tools as needed to answer the user's question.  You will be rewarded for taking some extra steps to find ALL the information requested from the user.
    """

    existing_messages = await retrieve_chat_history(
        username=payload.username, botname="slackbot-is"
    )

    # Always add the system message
    messages = [("system", system_prompt)]
    # Make the list of tuples format that LangChain expects
    for m in existing_messages:
        messages.append((m["role"], m["content"]))
    # add the current message
    messages += [
        ("user", payload.message),
        # placeholder for the agent scratchpad too.
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]

    # Make template, add the messages
    prompt = ChatPromptTemplate.from_messages(messages)
    # pprint(f"-------------Here is the current Prompt-----------------/n{prompt}")

    # Bind tools to llm
    llm_with_tools = llm.bind(
        functions=[format_tool_to_openai_function(t) for t in tools]
    )
    # use an OpenAI Functions compatible Tool Schema and Agent Scratchpad
    agent_schema = {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_functions(
            x["intermediate_steps"]
        ),
    }
    # Use LCEL syntax to build the agent
    agent = agent_schema | prompt | llm_with_tools | OpenAIFunctionsAgentOutputParser()
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Get the agent response object.
    a = await agent_executor.ainvoke({"input": payload.message})

    # append response and user question to history
    add_history_user_message = await insert_chat_history(
        username=payload.username,
        botname="slackbot-is",
        role="user",
        content=payload.message,
    )
    add_history_ai_message = await insert_chat_history(
        username=payload.username,
        botname="slackbot-is",
        role="ai",
        content=a["output"],
    )

    return {
        "username": payload.username,
        "context": payload.context,
        "contextType": payload.contextType,
        "message": a["output"],
        "sources": [],
    }

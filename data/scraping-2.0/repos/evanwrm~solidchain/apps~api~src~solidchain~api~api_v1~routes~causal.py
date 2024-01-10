from importlib.metadata import version
from typing import Any, List, Optional

from fastapi import APIRouter, Body, Depends
from fastapi.responses import StreamingResponse
from langchain.agents import initialize_agent
from langchain.agents.load_tools import get_all_tool_names, load_tools
from langchain.agents.tools import Tool
from langchain.chains import ConversationChain, VectorDBQA
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains.summarize import load_summarize_chain
from pydantic import BaseModel
from sqlalchemy.orm import Session

from solidchain.api.dependencies import get_db
from solidchain.configs.config import settings
from solidchain.models.vectorstore import VectorStore
from solidchain.schemas.agents import Agent, AgentTool
from solidchain.schemas.chains import SummarizeChainType
from solidchain.schemas.text_generation import (
    CausalGeneration,
    CausalModel,
    StreamingCausalGeneration,
)
from solidchain.schemas.vectorstore import VectorStore as VectorStoreSchema
from solidchain.utils import utils as sc_utils
from solidchain.utils.embeddings import get_embeddings_instance
from solidchain.utils.encoding import serialize_response
from solidchain.utils.llms import get_llm_instance
from solidchain.utils.vectorstores import get_vectorstore_instance

router = APIRouter()


@router.post("/generate", response_model=CausalGeneration)
def generate(
    *,
    text: str = Body(),
    modelName: CausalModel = Body("text-curie-001"),
    temperature: float = Body(0.7),
    maxTokens: int = Body(1024),
    streaming: bool = Body(False),
) -> Any:
    llm_cls = get_llm_instance(llm_type=modelName)
    llm = llm_cls(
        model_name=modelName,
        temperature=temperature,
        max_tokens=maxTokens,
        streaming=streaming,
        openai_api_key=settings.OPENAI_API_KEY,
    )

    if streaming:

        def streaming_response():
            try:
                generator = llm.stream(text)
                for output in generator:
                    generation = StreamingCausalGeneration(
                        text=output["choices"][0]["text"]
                    )
                    yield generation.json()
            except Exception as e:
                generation = StreamingCausalGeneration(text=llm(text))
                yield generation.json()

        return StreamingResponse(streaming_response())
    else:
        output = llm(text)
        generation = CausalGeneration(
            text=output.strip(),
        )
        return generation


@router.post("/qa", response_model=CausalGeneration)
def qa(
    *,
    db: Session = Depends(get_db),
    text: str = Body(),
    modelName: CausalModel = Body("text-curie-001"),
    temperature: float = Body(0.7),
    maxTokens: int = Body(1024),
    agent: Agent = Body("zero-shot-react-description"),
    agentPath: str = Body(None),
    agentTools: List[str] = Body(["serpapi", "llm-math"]),
    chainType: SummarizeChainType = Body("stuff"),
) -> Any:
    llm_cls = get_llm_instance(llm_type=modelName)
    llm = llm_cls(
        model_name=modelName,
        temperature=temperature,
        max_tokens=maxTokens,
        openai_api_key=settings.OPENAI_API_KEY,
    )

    if agent or agentPath:
        base_tools = set(agentTools) & set(get_all_tool_names())
        vector_tools = set(agentTools) - base_tools
        tools = load_tools(
            base_tools, llm=llm, serpapi_api_key=settings.SERPAPI_API_KEY
        )
        if vector_tools:
            vectorstores: List[VectorStoreSchema] = (
                db.query(VectorStore)
                .filter(VectorStore.vectorstoreId.in_(vector_tools))
                .all()
            )
            for vectorstore_data in vectorstores:
                embeddings = get_embeddings_instance(vectorstore_data.embeddingsType)
                vectorstore = get_vectorstore_instance(
                    vectorstore_data.vectorDb,
                    persist_directory=vectorstore_data.index.path,
                    embedding_function=embeddings,
                )
                vectorstore_qachain = VectorDBQA.from_chain_type(
                    llm=llm, chain_type=chainType, vectorstore=vectorstore
                )
                tool = Tool(
                    name=vectorstore_data.name,
                    description=vectorstore_data.description,
                    func=vectorstore_qachain.run,
                )
                tools.append(tool)

        if agent:
            agent_executor = initialize_agent(
                tools=tools,
                llm=llm,
                agent=agent,
                max_iterations=5,
                early_stopping_method="generate",
                return_intermediate_steps=True,
            )
            output = agent_executor(text)
        elif agentPath:
            agent_executor = initialize_agent(
                tools=tools,
                llm=llm,
                agent_path=agentPath,
                max_iterations=5,
                early_stopping_method="generate",
                return_intermediate_steps=True,
            )
        output = agent_executor(text)
    else:
        output = llm(text)

    generation = CausalGeneration(
        text=output["output"].strip(), steps=output["intermediate_steps"]
    )
    return generation


@router.post("/summarize", response_model=CausalGeneration)
def summarize(
    *,
    text: str = Body(),
    modelName: CausalModel = Body("text-curie-001"),
    temperature: float = Body(0.7),
    maxTokens: int = Body(1024),
    chainType: SummarizeChainType = Body("stuff"),
) -> Any:
    llm_cls = get_llm_instance(llm_type=modelName)
    llm = llm_cls(
        model_name=modelName,
        temperature=temperature,
        max_tokens=maxTokens,
        openai_api_key=settings.OPENAI_API_KEY,
    )
    chain = load_summarize_chain(llm=llm, chain_type=chainType)
    output = chain.run(text)

    generation = CausalGeneration(
        text=output.strip(),
    )
    return generation


@router.post("/conversational", response_model=CausalGeneration)
def conversational(
    *,
    text: str = Body(),
    modelName: CausalModel = Body("gpt-3.5-turbo"),
    temperature: float = Body(0.7),
    maxTokens: int = Body(1024),
) -> Any:
    llm_cls = get_llm_instance(llm_type=modelName)
    llm = llm_cls(
        model_name=modelName,
        temperature=temperature,
        max_tokens=maxTokens,
        openai_api_key=settings.OPENAI_API_KEY,
    )
    chain = ConversationChain(llm=llm, memory=ConversationBufferMemory())
    output = chain.run(text)

    generation = CausalGeneration(
        text=output.strip(),
    )
    return generation

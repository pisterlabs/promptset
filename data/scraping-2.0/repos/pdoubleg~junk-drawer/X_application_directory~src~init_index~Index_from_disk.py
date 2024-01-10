from llama_index import (
    ServiceContext,
    StorageContext,
    LLMPredictor,
    load_index_from_storage,
)
from llama_index.question_gen.llm_generators import LLMQuestionGenerator
from llama_index.question_gen.prompts import DEFAULT_SUB_QUESTION_PROMPT_TMPL
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.tools import QueryEngineTool, ToolMetadata
from langchain.chat_models import ChatOpenAI


def get_llm_predictor(temperature=0):
    return LLMPredictor(ChatOpenAI(temperature=temperature, model="gpt-3.5-turbo"))


service_context = ServiceContext.from_defaults(llm_predictor=get_llm_predictor())

index = load_index_from_storage(
    StorageContext.from_defaults(persist_dir="./_policy_index_metadatas"),
)

engine = index.as_query_engine(similarity_top_k=10)

question_gen = LLMQuestionGenerator.from_defaults(
    service_context=service_context,
    prompt_template_str="""
        Follow the example, but instead of giving a question, always prefix the question 
        with: 'By first identifying and quoting the most relevant sources, '. 
        """
    + DEFAULT_SUB_QUESTION_PROMPT_TMPL,
)

final_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=[
        QueryEngineTool(
            query_engine=engine,
            metadata=ToolMetadata(
                name="homeowners_policy_documents",
                description="insurance policy contract detailing coverage provisions.",
            ),
        )
    ],
    service_context=service_context,
    use_async=False,
)

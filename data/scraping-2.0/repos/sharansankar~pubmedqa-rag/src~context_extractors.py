import attr
from typing import Optional, List

from data_model import PromptResponseAttributes, QueryContextType, QueryResult
from llm_agent import OpenAILLMChatClient
from prompt_generator import PromptGenerator
from prompt_response_attrs_utils import get_end_user_prompt, get_query_context_extracted_values
from vector_db import HuggingFaceDatasetStore


@attr.s
class ExtractorContext:
  dataset_store: HuggingFaceDatasetStore = attr.ib()
  self_querying_prompt_generator: PromptGenerator = attr.ib()
  enabled_extraction_stages: List[QueryContextType] = attr.ib()
  llm_agent: OpenAILLMChatClient = attr.ib()
  self_querying_num_topics: int = attr.ib(default=3)


def dataset_context_extractor_from_prompt(
  pr_attrs: PromptResponseAttributes,
  extractor_context: ExtractorContext
) -> QueryResult:
  query_list = [
    get_end_user_prompt(pr_attrs)
  ]
  if QueryContextType.PUBMED_TOPIC_SELF_QUERY in extractor_context.enabled_extraction_stages:
    extracted_vals = get_query_context_extracted_values(pr_attrs, QueryContextType.PUBMED_TOPIC_SELF_QUERY)
    if len(extracted_vals) > 1:
      query_list = extracted_vals

  dataset_store = extractor_context.dataset_store
  return dataset_store.query_dataset(query_list)


def dataset_self_querying_extractor(
  pr_attrs: PromptResponseAttributes,
  extractor_context: ExtractorContext
) -> QueryResult:
  # # 1 get prompt from pr attrs
  # end_user_prompt = get_end_user_prompt(pr_attrs)

  # 2 build LLM self-querying prompt
  prompt_generator = extractor_context.self_querying_prompt_generator
  llm_agent = extractor_context.llm_agent

  # 3 get LLM self-querying response
  llm_response = llm_agent.get_response_from_prompt(prompt_generator.build_prompt(pr_attrs))

  # 4 parse out LLM response
  llm_response_text = llm_response.text_response
  llm_response_topics = llm_response_text.split(',')

  return QueryResult(
    raw_text_response=llm_response_text,
    extracted_values=llm_response_topics,
    metadata=llm_response.metadata
  )


QUERY_CONTEXT_TYPE_TO_EXTRACTOR = {
  QueryContextType.PUBMED_CONTEXT: dataset_context_extractor_from_prompt,
  QueryContextType.PUBMED_TOPIC_SELF_QUERY: dataset_self_querying_extractor
}
import time

from context_extractors import ExtractorContext, QUERY_CONTEXT_TYPE_TO_EXTRACTOR
from data_model import SessionAttriubtes, PromptResponseAttributes, EndUserPrompt, QueryContextType, HumanFeedback
from llm_agent import OpenAILLMChatClient
from prompt_generator import PromptGenerator
from prompt_response_attrs_utils import get_query_context_extracted_values
from vector_db import HuggingFaceDatasetStore
from configs import SessionConfig


class SessionBuilder:
  def _gen_session_id(self) -> int:
    return int(time.time())

  def _build_session(self, ):
    session = SessionAttriubtes(
      session_id=self._gen_session_id()
    )
    self.current_session = session

  def __init__(self, session_config: SessionConfig):
    self._build_session()
    self.data_store = HuggingFaceDatasetStore(session_config.data_store_config)
    self.llm_agent = OpenAILLMChatClient(session_config.llm_client_config)

    if QueryContextType.PUBMED_CONTEXT in session_config.extraction_stages:
      self.context_prompt_generator = PromptGenerator(QueryContextType.PUBMED_CONTEXT)

    if QueryContextType.PUBMED_TOPIC_SELF_QUERY in session_config.extraction_stages:
      self.self_querying_prompt_generator = PromptGenerator(QueryContextType.PUBMED_TOPIC_SELF_QUERY)

    self.last_pr_attrs = None
    self.enabled_extraction_stages = session_config.extraction_stages
    self.debugging_enabled = session_config.debugging

  def create_prompt_response_attrs_with_prompt(self, prompt: str):
    return PromptResponseAttributes(
      end_user_prompt=EndUserPrompt(
        prompt_text=prompt
      )
    )

  def _update_session_state(self, pr_attrs: PromptResponseAttributes):
    self.latest_pr_attrs = pr_attrs
    self.current_session.PromptResponseAttributesList.append(pr_attrs)

  def process_user_prompt(self, prompt: str) -> str:
    # 1 create prompt obj with end user prompt
    pr_attrs = self.create_prompt_response_attrs_with_prompt(prompt)

    # 2 hydrate context for prompt
    extractor_context = ExtractorContext(
        dataset_store=self.data_store,
        self_querying_prompt_generator=self.self_querying_prompt_generator,
        enabled_extraction_stages=self.enabled_extraction_stages,
        llm_agent=self.llm_agent
      )
    for query_context_type in self.enabled_extraction_stages:
      extractor = QUERY_CONTEXT_TYPE_TO_EXTRACTOR.get(query_context_type)
      pr_attrs.context_querying_results[query_context_type] = extractor(
        pr_attrs=pr_attrs,
        extractor_context=extractor_context
      )
      if self.debugging_enabled:
        print(f"--------------- {str(query_context_type)} ---------------")
        print(get_query_context_extracted_values(pr_attrs, query_context_type))

    # 3 Build LLM Agent Prompt
    llm_agent_prompt = self.context_prompt_generator.build_prompt(pr_attrs)
    if self.debugging_enabled:
      print("--------------- LLM Agent Prompt ---------------")
      print(llm_agent_prompt)

    # 4 Get Prompt
    pr_attrs.llm_prompt_response = self.llm_agent.get_response_from_prompt(llm_agent_prompt)
    if self.debugging_enabled:
      print("--------------- LLM Response ---------------")
      print(pr_attrs.llm_prompt_response.text_response)

    # 5 Update Session state
    self._update_session_state(pr_attrs)

    return pr_attrs.llm_prompt_response.text_response

  def response_is_upvoted(self):
    self.latest_pr_attrs.human_feedback = HumanFeedback(is_upvoted=True)

  def enable_debugging(self):
    self.debugging_enabled = True

  def disable_debugging(self):
    self.debugging_enabled = False


if __name__ == "__main__":
  from configs import DEMO_SESSION_CONFIG

  session = SessionBuilder(DEMO_SESSION_CONFIG)
  session.enable_debugging()
  session.process_user_prompt("is diabetes a medical condition?")
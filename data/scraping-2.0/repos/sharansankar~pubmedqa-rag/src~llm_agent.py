from dotenv import load_dotenv
import openai
import os
from typing import List, Dict, Optional

from data_model import LLMPromptResponse


def init_open_ai():
  load_dotenv(dotenv_path="../.env")
  openai.organization = os.getenv("OPENAI_ORG_KEY")
  openai.api_key = os.getenv("OPENAI_API_KEY")
  return


class OpenAILLMChatClient:
  def __init__(self, config: "LLMConfig"):
    self.model_name = config.model_name.value
    self.max_tokens = config.max_tokens
    self.temperature = config.temperature

    if openai.organization is None or openai.api_key is None:
      init_open_ai()

  def _parse_llm_metadata_for_response(self, llm_metadata: Dict) -> str:
    choices = llm_metadata.get('choices', [])
    if len(choices) > 0:
      message = choices[0].get('message', None)
      if message is not None:
        return message.get('content',"")

  def _parse_llm_response(self, llm_response) -> LLMPromptResponse:
    metadata = llm_response.to_dict_recursive()
    return LLMPromptResponse(
      text_response=self._parse_llm_metadata_for_response(metadata),
      metadata=metadata
    )

  def get_response_from_prompt(
    self,
    prompt: str,
    prev_messages: Optional[List[Dict[str, str]]] = None,
    role='user'
  ) -> LLMPromptResponse:
    if prev_messages:
      prev_messages.append({
          'role': role,
          'content': prompt
      })
    else:
      prev_messages = [
        {
          'role': role,
          'content': prompt
        }
      ]
    response = openai.ChatCompletion.create(
      model=self.model_name,
      messages=prev_messages,
      max_tokens=self.max_tokens,
      temperature=self.temperature
    )
    return self._parse_llm_response(response)


if __name__ == "__main__":
  from configs import DEMO_LLM_CONFIG
  llm_client = OpenAILLMChatClient(DEMO_LLM_CONFIG)
  response = llm_client.get_response_from_prompt("hello what is your name?")
  print(response)

# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tools and utils for Gmail"""

from langchain.prompts import ChatPromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

from config import OPENAI_API_KEY
from common.utils.logging_handler import Logger

Logger = Logger.get_logger(__file__)

def compose_email(
    prompt: str, email_template: str, variables: dict = None) -> dict:
  """
    Compose a new email based on the prompt and template.
    Args:
        prompt: String
        email_template: String
        variables: Dict
    Returns:
        {subject, message}
  """
  # Generate email body.
  subject_template = """
      Create a subject for the following email: {email_body} \n\n

      """
  email_body_prompt = ChatPromptTemplate.from_template(email_template)
  subject_prompt = ChatPromptTemplate.from_template(subject_template)

  # TODO: Add options to use VertexAI and other models.
  llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

  # Generate email content.
  llm_chain = LLMChain(prompt=email_body_prompt, llm=llm)
  message = llm_chain.run(email_body=prompt, **variables)

  # Generate email subject.
  llm_chain = LLMChain(prompt=subject_prompt, llm=llm)
  subject = llm_chain.run(email_body=message)

  return {
    "subject": subject,
    "message": message,
  }

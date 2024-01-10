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
from langchain.tools.gmail.utils import build_resource_service
from langchain.agents.agent_toolkits import GmailToolkit
from common.utils.logging_handler import Logger
from utils.google_credential import get_google_credential

Logger = Logger.get_logger(__file__)

def send_email(recipient, subject, message):
  """
    Send an email to a recipient with message.
    Args:
        Bearer Token: String
    Returns:
        Decoded Token and User type: Dict
  """
  cred = get_google_credential()
  api_resource = build_resource_service(credentials=cred)
  toolkit = GmailToolkit(api_resource=api_resource)
  tools = toolkit.get_tools()

  email_content = {
    "to": recipient,
    "subject": subject,
    "message": message
  }

  # Find the GmailSendMessage tool from the toolkit list.
  send_message_tool = None
  for tool in tools:
    if tool.__class__.__name__ == "GmailSendMessage":
      send_message_tool = tool

  if not send_message_tool:
    raise RuntimeError(
        "Unable to locate 'GmailSendMessage' from LangChain toolkit.")

  result = send_message_tool.invoke(input=email_content)
  return result

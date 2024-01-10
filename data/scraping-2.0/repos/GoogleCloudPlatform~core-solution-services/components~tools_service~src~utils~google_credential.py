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

# TODO: Remove dependency from langchain.
from langchain.tools.gmail.utils import get_gmail_credentials
from google.cloud import secretmanager

from common.utils.logging_handler import Logger
from common.utils.secrets import get_secret


Logger = Logger.get_logger(__file__)

sm_client = secretmanager.SecretManagerServiceClient()
SCOPES = [
  "https://mail.google.com/"
]

def get_google_credential():
  # TODO: Explore other ways to get credentials without the need of OAuth token.
  # Write credential files to local tmp as the temporary approach.
  oauth_token = get_secret("tools-gmail-oauth-token")
  token_file_path = "/tmp/oauth_token.json"
  client_secrets = get_secret("tools-gmail-client-secrets")
  client_secrets_file_path = "/tmp/client_secrets.json"

  with open(token_file_path, "w", encoding="utf-8") as file:
    file.write(oauth_token)
  with open(client_secrets_file_path, "w", encoding="utf-8") as file:
    file.write(client_secrets)

  credentials = get_gmail_credentials(
      token_file=token_file_path,
      scopes=SCOPES,
      client_secrets_file=client_secrets_file_path,
  )
  return credentials

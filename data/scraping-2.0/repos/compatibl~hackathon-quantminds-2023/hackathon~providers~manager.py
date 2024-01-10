# Copyright (C) 2023-present The Project Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from hackathon.models.ai_models import AIProvider
from hackathon.providers.base_provider import BaseProvider
from hackathon.providers.fireworks_provider import FireworksProvider
from hackathon.providers.openai_provider import OpenAIProvider
from hackathon.providers.replicate_provider import ReplicateProvider


def get_provider(ai_provider: AIProvider, api_key: str) -> BaseProvider:
    return {
        AIProvider.REPLICATE: ReplicateProvider,
        AIProvider.FIREWORKS: FireworksProvider,
        AIProvider.OPENAI: OpenAIProvider,
    }[ai_provider](api_key)

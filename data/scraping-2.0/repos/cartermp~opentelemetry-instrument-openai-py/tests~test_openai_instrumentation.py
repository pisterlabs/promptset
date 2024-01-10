# Copyright Phillip Carter
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import openai
import math
from openai import util
from unittest import mock
from unittest.mock import create_autospec
from openai.api_resources.abstract.engine_api_resource import EngineAPIResource
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.test.test_base import TestBase


class MockChatCompletion(EngineAPIResource):
    @classmethod
    def create(cls, *args, **kwargs):
        return {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "\n\nOpenTelemetry is easy to use",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 9, "completion_tokens": 10, "total_tokens": 19},
        }


class MockCompletion(EngineAPIResource):
    @classmethod
    def create(cls, *args, **kwargs):
        return {
            "id": "cmpl-123",
            "object": "text_completion",
            "created": 1677652288,
            "choices": [
                {
                    "text": "OpenTelemetry is easy to use",
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 9, "completion_tokens": 10, "total_tokens": 19},
        }


class MockEmbedding(EngineAPIResource):
    @classmethod
    def create(cls, *args, **kwargs):
        data = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [
                        0.0023064255,
                        -0.009327292,
                        -0.0028842222,
                        # Eliding 1536 other floats
                    ],
                    "index": 0,
                }
            ],
            "model": "text-embedding-ada-002",
            "usage": {
                "prompt_tokens": 8,
                "total_tokens": 8,
            },
        }
        return util.convert_to_openai_object(data)


class MockEdit(EngineAPIResource):
    @classmethod
    def create(cls, *args, **kwargs):
        return {
            "object": "edit",
            "created": 1589478378,
            "choices": [
                {
                    "text": "What day of the week is it?",
                    "index": 0,
                }
            ],
            "usage": {"prompt_tokens": 25, "completion_tokens": 32, "total_tokens": 57},
        }


class MockModeration(EngineAPIResource):
    @classmethod
    def create(cls, *args, **kwargs):
        return {
            "id": "modr-5MWoLO",
            "model": "text-moderation-001",
            "results": [
                {
                    "categories": {
                        "hate": False,
                        "hate/threatening": True,
                        "self-harm": False,
                        "sexual": False,
                        "sexual/minors": False,
                        "violence": True,
                        "violence/graphic": False,
                    },
                    "category_scores": {
                        "hate": 0.22714105248451233,
                        "hate/threatening": 0.4132447838783264,
                        "self-harm": 0.005232391878962517,
                        "sexual": 0.01407341007143259,
                        "sexual/minors": 0.0038522258400917053,
                        "violence": 0.9223177433013916,
                        "violence/graphic": 0.036865197122097015,
                    },
                    "flagged": True,
                }
            ],
        }


class MockImage(EngineAPIResource):
    @classmethod
    def create(cls, *args, **kwargs):
        return {
            "created": 1589478378,
            "data": [{"url": "example.com"}, {"url": "gooddeals.com"}],
        }

    @classmethod
    def create_edit(cls, *args, **kwargs):
        return {
            "created": 1589478378,
            "data": [{"url": "example.com"}, {"url": "gooddeals.com"}],
        }

    @classmethod
    def create_variation(cls, *args, **kwargs):
        return {
            "created": 1589478378,
            "data": [{"url": "example.com"}, {"url": "gooddeals.com"}],
        }


class MockAudio(EngineAPIResource):
    @classmethod
    def transcribe(cls, *args, **kwargs):
        return {
            "text": "It's 'a me, Mario!",
        }

    @classmethod
    def translate(cls, *args, **kwargs):
        return {
            "text": "It's 'a me, Mario!",
        }


class TestOpenAIInstrumentation(TestBase):
    def _assert_spans(self, num_spans: int):
        finished_spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(num_spans, len(finished_spans))
        if num_spans == 0:
            return None
        if num_spans == 1:
            return finished_spans[0]
        return finished_spans

    def _assert_api_attributes(self, span):
        self.assertEqual(
            span.attributes["openai.api_base"],
            "https://api.openai.com/v1",
        )
        self.assertEqual(
            span.attributes["openai.api_type"],
            "open_ai",
        )
        self.assertEqual(
            span.attributes["openai.api_version"],
            "None",
        )

    def _call_chat(self):
        return openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "tell me a joke about opentelemetry"}
            ],
            temperature=0.0,
            max_tokens=150,
            user="test",
        )
    
    def test_instrument_chat_no_message_inputs(self):
        mock_chat_completion = create_autospec(openai.ChatCompletion)
        mock_chat_completion.create = MockChatCompletion.create
        with mock.patch("openai.ChatCompletion", new=mock_chat_completion):
            OpenAIInstrumentor().instrument(suppress_input_content = True)

            result = self._call_chat()

            span = self._assert_spans(1)
            name = "openai.chat"
            self.assertEqual(span.name, name)
            self.assertEqual(span.attributes[f"{name}.model"], "gpt-3.5-turbo")
            self.assertNotIn(
                f"{name}.messages.0.role",
                span.attributes,
                f"Unexpected attribute found: {name}.messages.0.role",
            )
            self.assertNotIn(
                f"{name}.messages.0.content",
                span.attributes,
                f"Unexpected attribute found: {name}.messages.0.content",
            )
            self.assertEqual(span.attributes[f"{name}.temperature"], 0.0)
            self.assertEqual(span.attributes[f"{name}.top_p"], 1.0)
            self.assertEqual(span.attributes[f"{name}.n"], 1)
            self.assertEqual(span.attributes[f"{name}.stream"], False)
            self.assertEqual(span.attributes[f"{name}.stop"], "")
            self.assertEqual(span.attributes[f"{name}.max_tokens"], 150)
            self.assertEqual(span.attributes[f"{name}.presence_penalty"], 0.0)
            self.assertEqual(span.attributes[f"{name}.frequency_penalty"], 0.0)
            self.assertEqual(span.attributes[f"{name}.logit_bias"], "")
            self.assertEqual(span.attributes[f"{name}.user"], "test")

            self.assertEqual(span.attributes[f"{name}.response.id"], result["id"])
            self.assertEqual(
                span.attributes[f"{name}.response.object"], result["object"]
            )
            self.assertEqual(
                span.attributes[f"{name}.response.created"], result["created"]
            )
            self.assertEqual(
                span.attributes[f"{name}.response.choices.0.message.role"],
                result["choices"][0]["message"]["role"],
            )
            self.assertEqual(
                span.attributes[f"{name}.response.choices.0.message.content"],
                result["choices"][0]["message"]["content"],
            )
            self.assertEqual(
                span.attributes[f"{name}.response.choices.0.finish_reason"],
                result["choices"][0]["finish_reason"],
            )
            self.assertEqual(
                span.attributes[f"{name}.response.usage.prompt_tokens"],
                result["usage"]["prompt_tokens"],
            )
            self.assertEqual(
                span.attributes[f"{name}.response.usage.completion_tokens"],
                result["usage"]["completion_tokens"],
            )
            self.assertEqual(
                span.attributes[f"{name}.response.usage.total_tokens"],
                result["usage"]["total_tokens"],
            )
            
            self._assert_api_attributes(span)

            OpenAIInstrumentor().uninstrument()

    
    def test_instrument_chat_no_message_content(self):
        mock_chat_completion = create_autospec(openai.ChatCompletion)
        mock_chat_completion.create = MockChatCompletion.create
        with mock.patch("openai.ChatCompletion", new=mock_chat_completion):
            OpenAIInstrumentor().instrument(suppress_response_data = True)

            result = self._call_chat()

            span = self._assert_spans(1)
            name = "openai.chat"
            self.assertEqual(span.name, name)
            self.assertEqual(span.attributes[f"{name}.model"], "gpt-3.5-turbo")
            self.assertEqual(
                span.attributes[f"{name}.messages.0.role"],
                "user",
            )
            self.assertEqual(
                span.attributes[f"{name}.messages.0.content"],
                "tell me a joke about opentelemetry",
            )
            self.assertEqual(span.attributes[f"{name}.temperature"], 0.0)
            self.assertEqual(span.attributes[f"{name}.top_p"], 1.0)
            self.assertEqual(span.attributes[f"{name}.n"], 1)
            self.assertEqual(span.attributes[f"{name}.stream"], False)
            self.assertEqual(span.attributes[f"{name}.stop"], "")
            self.assertEqual(span.attributes[f"{name}.max_tokens"], 150)
            self.assertEqual(span.attributes[f"{name}.presence_penalty"], 0.0)
            self.assertEqual(span.attributes[f"{name}.frequency_penalty"], 0.0)
            self.assertEqual(span.attributes[f"{name}.logit_bias"], "")
            self.assertEqual(span.attributes[f"{name}.user"], "test")

            self.assertEqual(span.attributes[f"{name}.response.id"], result["id"])
            self.assertEqual(
                span.attributes[f"{name}.response.object"], result["object"]
            )
            self.assertEqual(
                span.attributes[f"{name}.response.created"], result["created"]
            )
            self.assertNotIn(
                f"{name}.response.choices.0.message.role",
                span.attributes,
                f"Unexpected attribute found: {name}.response.choices.0.message.role",
            )
            self.assertNotIn(
                f"{name}.response.choices.0.message.content",
                span.attributes,
                f"Unexpected attribute found: {name}.response.choices.0.message.content",
            )
            self.assertNotIn(
                f"{name}.response.choices.0.finish_reason",
                span.attributes,
                f"Unexpected attribute found: {name}.response.choices.0.finish_reason",
            )
            self.assertEqual(
                span.attributes[f"{name}.response.usage.prompt_tokens"],
                result["usage"]["prompt_tokens"],
            )
            self.assertEqual(
                span.attributes[f"{name}.response.usage.completion_tokens"],
                result["usage"]["completion_tokens"],
            )
            self.assertEqual(
                span.attributes[f"{name}.response.usage.total_tokens"],
                result["usage"]["total_tokens"],
            )
            
            self._assert_api_attributes(span)

            OpenAIInstrumentor().uninstrument()


    def test_instrument_chat(self):
        mock_chat_completion = create_autospec(openai.ChatCompletion)
        mock_chat_completion.create = MockChatCompletion.create
        with mock.patch("openai.ChatCompletion", new=mock_chat_completion):
            OpenAIInstrumentor().instrument()

            result = self._call_chat()

            span = self._assert_spans(1)
            name = "openai.chat"
            self.assertEqual(span.name, name)
            self.assertEqual(span.attributes[f"{name}.model"], "gpt-3.5-turbo")
            self.assertEqual(
                span.attributes[f"{name}.messages.0.role"],
                "user",
            )
            self.assertEqual(
                span.attributes[f"{name}.messages.0.content"],
                "tell me a joke about opentelemetry",
            )
            self.assertEqual(span.attributes[f"{name}.temperature"], 0.0)
            self.assertEqual(span.attributes[f"{name}.top_p"], 1.0)
            self.assertEqual(span.attributes[f"{name}.n"], 1)
            self.assertEqual(span.attributes[f"{name}.stream"], False)
            self.assertEqual(span.attributes[f"{name}.stop"], "")
            self.assertEqual(span.attributes[f"{name}.max_tokens"], 150)
            self.assertEqual(span.attributes[f"{name}.presence_penalty"], 0.0)
            self.assertEqual(span.attributes[f"{name}.frequency_penalty"], 0.0)
            self.assertEqual(span.attributes[f"{name}.logit_bias"], "")
            self.assertEqual(span.attributes[f"{name}.user"], "test")

            self.assertEqual(span.attributes[f"{name}.response.id"], result["id"])
            self.assertEqual(
                span.attributes[f"{name}.response.object"], result["object"]
            )
            self.assertEqual(
                span.attributes[f"{name}.response.created"], result["created"]
            )
            self.assertEqual(
                span.attributes[f"{name}.response.choices.0.message.role"],
                result["choices"][0]["message"]["role"],
            )
            self.assertEqual(
                span.attributes[f"{name}.response.choices.0.message.content"],
                result["choices"][0]["message"]["content"],
            )
            self.assertEqual(
                span.attributes[f"{name}.response.choices.0.finish_reason"],
                result["choices"][0]["finish_reason"],
            )
            self.assertEqual(
                span.attributes[f"{name}.response.usage.prompt_tokens"],
                result["usage"]["prompt_tokens"],
            )
            self.assertEqual(
                span.attributes[f"{name}.response.usage.completion_tokens"],
                result["usage"]["completion_tokens"],
            )
            self.assertEqual(
                span.attributes[f"{name}.response.usage.total_tokens"],
                result["usage"]["total_tokens"],
            )
            
            self._assert_api_attributes(span)

            OpenAIInstrumentor().uninstrument()

    def test_instrument_completion(self):
        mock_completion = create_autospec(openai.Completion)
        mock_completion.create = MockCompletion.create
        with mock.patch("openai.Completion", new=mock_completion):
            OpenAIInstrumentor().instrument()

            result = openai.Completion.create(
                model="text-davinci-003",
                prompt="tell me a joke about opentelemetry",
                user="test",
            )

            span = self._assert_spans(1)
            name = "openai.completion"
            self.assertEqual(span.name, name)
            self.assertEqual(span.attributes[f"{name}.model"], "text-davinci-003")
            self.assertEqual(
                span.attributes[f"{name}.prompt"],
                "tell me a joke about opentelemetry",
            )
            self.assertEqual(span.attributes[f"{name}.temperature"], 1.0)
            self.assertEqual(span.attributes[f"{name}.top_p"], 1.0)
            self.assertEqual(span.attributes[f"{name}.n"], 1)
            self.assertEqual(span.attributes[f"{name}.stream"], False)
            self.assertEqual(span.attributes[f"{name}.stop"], "")
            self.assertEqual(span.attributes[f"{name}.max_tokens"], math.inf)
            self.assertEqual(span.attributes[f"{name}.presence_penalty"], 0.0)
            self.assertEqual(span.attributes[f"{name}.frequency_penalty"], 0.0)
            self.assertEqual(span.attributes[f"{name}.logit_bias"], "")
            self.assertEqual(span.attributes[f"{name}.user"], "test")

            self.assertEqual(span.attributes[f"{name}.response.id"], result["id"])
            self.assertEqual(
                span.attributes[f"{name}.response.object"], result["object"]
            )
            self.assertEqual(
                span.attributes[f"{name}.response.created"], result["created"]
            )
            self.assertEqual(
                span.attributes[f"{name}.response.choices.0.text"],
                result["choices"][0]["text"],
            )
            self.assertEqual(
                span.attributes[f"{name}.response.choices.0.finish_reason"],
                result["choices"][0]["finish_reason"],
            )
            self.assertEqual(
                span.attributes[f"{name}.response.usage.prompt_tokens"],
                result["usage"]["prompt_tokens"],
            )
            self.assertEqual(
                span.attributes[f"{name}.response.usage.completion_tokens"],
                result["usage"]["completion_tokens"],
            )
            self.assertEqual(
                span.attributes[f"{name}.response.usage.total_tokens"],
                result["usage"]["total_tokens"],
            )

            self._assert_api_attributes(span)

            OpenAIInstrumentor().uninstrument()

    def test_instrument_embedding(self):
        mock_embedding = create_autospec(openai.Embedding)
        mock_embedding.create = MockEmbedding.create
        with mock.patch("openai.Embedding", new=mock_embedding):
            OpenAIInstrumentor().instrument()

            result = openai.Embedding.create(
                model="text-embedding-ada-002",
                input="The food was delicious and the waiter...",
                user="test",
            )

            span = self._assert_spans(1)
            name = "openai.embedding"
            self.assertEqual(span.name, name)
            self.assertEqual(span.attributes[f"{name}.model"], "text-embedding-ada-002")
            self.assertEqual(span.attributes[f"{name}.input_count"], 40)
            self.assertEqual(span.attributes[f"{name}.user"], "test")
            # TODO: lol this doesn't work
            # the patching required for how openai's thing works here is too much for my brain
            # print(result)
            # self.assertEqual(
            #     span.attributes[f"{name}.response.embeddings_count"], len(result.data)
            # )
            # self.assertEqual(
            #     span.attributes[f"{name}.response.usage.promt_tokens"],
            #     result.usage.prompt_token,
            # )
            # self.assertEqual(
            #     span.attributes[f"{name}.response.usage.total_tokens"],
            #     result.usage.total_tokens,
            # )
            
            self._assert_api_attributes(span)

            OpenAIInstrumentor().uninstrument()

    def test_instrument_edit(self):
        mock_edit = create_autospec(openai.Edit)
        mock_edit.create = MockEdit.create
        with mock.patch("openai.Edit", new=mock_edit):
            OpenAIInstrumentor().instrument()

            result = openai.Edit.create(
                model="text-davinci-edit-001",
                input="What day of the wek is it?",
                instruction="Fix the spelling mistakes.",
            )

            span = self._assert_spans(1)
            name = "openai.edit"
            self.assertEqual(span.name, name)
            self.assertEqual(span.attributes[f"{name}.model"], "text-davinci-edit-001")
            self.assertEqual(
                span.attributes[f"{name}.input"], "What day of the wek is it?"
            )
            self.assertEqual(
                span.attributes[f"{name}.instruction"], "Fix the spelling mistakes."
            )
            self.assertEqual(
                span.attributes[f"{name}.response.object"], result["object"]
            )
            self.assertEqual(
                span.attributes[f"{name}.response.created"], result["created"]
            )
            self.assertEqual(
                span.attributes[f"{name}.response.choices.0.text"],
                result["choices"][0]["text"],
            )
            self.assertEqual(
                span.attributes[f"{name}.response.usage.prompt_tokens"],
                result["usage"]["prompt_tokens"],
            )
            self.assertEqual(
                span.attributes[f"{name}.response.usage.completion_tokens"],
                result["usage"]["completion_tokens"],
            )
            self.assertEqual(
                span.attributes[f"{name}.response.usage.total_tokens"],
                result["usage"]["total_tokens"],
            )

            self._assert_api_attributes(span)

            OpenAIInstrumentor().uninstrument()

    def test_instrument_moderation(self):
        mock_moderation = create_autospec(openai.Moderation)
        mock_moderation.create = MockModeration.create
        with mock.patch("openai.Moderation", new=mock_moderation):
            OpenAIInstrumentor().instrument()

            result = openai.Moderation.create(
                model="text-moderation-latest",
                input="I want to kill them.",
            )

            span = self._assert_spans(1)
            name = "openai.moderation"

            self.assertEqual(span.name, name)
            self.assertEqual(span.attributes[f"{name}.model"], "text-moderation-latest")
            self.assertEqual(span.attributes[f"{name}.input"], "I want to kill them.")
            self.assertEqual(span.attributes[f"{name}.response.id"], result["id"])
            self.assertEqual(
                span.attributes[f"{name}.response.results.0.categories.hate"],
                result["results"][0]["categories"]["hate"],
            )
            self.assertEqual(
                span.attributes[f"{name}.response.results.0.categories.hate/threatening"],
                result["results"][0]["categories"]["hate/threatening"],
            )
            self.assertEqual(
                span.attributes[f"{name}.response.results.0.categories.self-harm"],
                result["results"][0]["categories"]["self-harm"],
            )
            self.assertEqual(
                span.attributes[f"{name}.response.results.0.categories.sexual"],
                result["results"][0]["categories"]["sexual"],
            )
            self.assertEqual(
                span.attributes[f"{name}.response.results.0.categories.sexual/minors"],
                result["results"][0]["categories"]["sexual/minors"],
            )
            self.assertEqual(
                span.attributes[f"{name}.response.results.0.categories.violence"],
                result["results"][0]["categories"]["violence"],
            )
            self.assertEqual(
                span.attributes[f"{name}.response.results.0.categories.violence/graphic"],
                result["results"][0]["categories"]["violence/graphic"],
            )
            self.assertEqual(
                span.attributes[f"{name}.response.results.0.category_scores.hate"],
                result["results"][0]["category_scores"]["hate"],
            )
            self.assertEqual(
                span.attributes[
                    f"{name}.response.results.0.category_scores.hate/threatening"
                ],
                result["results"][0]["category_scores"]["hate/threatening"],
            )
            self.assertEqual(
                span.attributes[f"{name}.response.results.0.category_scores.self-harm"],
                result["results"][0]["category_scores"]["self-harm"],
            )
            self.assertEqual(
                span.attributes[f"{name}.response.results.0.category_scores.sexual"],
                result["results"][0]["category_scores"]["sexual"],
            )
            self.assertEqual(
                span.attributes[
                    f"{name}.response.results.0.category_scores.sexual/minors"
                ],
                result["results"][0]["category_scores"]["sexual/minors"],
            )
            self.assertEqual(
                span.attributes[f"{name}.response.results.0.category_scores.violence"],
                result["results"][0]["category_scores"]["violence"],
            )
            self.assertEqual(
                span.attributes[
                    f"{name}.response.results.0.category_scores.violence/graphic"
                ],
                result["results"][0]["category_scores"]["violence/graphic"],
            )
            self.assertEqual(
                span.attributes[f"{name}.response.results.0.flagged"],
                result["results"][0]["flagged"],
            )

            self._assert_api_attributes(span)

            OpenAIInstrumentor().uninstrument()

    def test_instrument_image_generate(self):
        mock_image = create_autospec(openai.Image)
        mock_image.create = MockImage.create
        with mock.patch("openai.Image", new=mock_image):
            OpenAIInstrumentor().instrument()

            result = openai.Image.create(
                prompt="A cute baby sea otter",
                n=2,
            )

            span = self._assert_spans(1)

            name = "openai.image.generate"
            self.assertEqual(span.name, name)
            self.assertEqual(span.attributes[f"{name}.prompt"], "A cute baby sea otter")
            self.assertEqual(span.attributes[f"{name}.n"], 2)
            self.assertEqual(span.attributes[f"{name}.size"], "1024x1024")
            self.assertEqual(span.attributes[f"{name}.response_format"], "url")
            self.assertEqual(
                span.attributes[f"{name}.response.created"], result["created"]
            )
            self.assertEqual(
                span.attributes[f"{name}.response.data.0.url"], result["data"][0]["url"]
            )
            self.assertEqual(
                span.attributes[f"{name}.response.data.1.url"], result["data"][1]["url"]
            )

            self._assert_api_attributes(span)

            OpenAIInstrumentor().uninstrument()

    def test_instrument_image_edit(self):
        mock_image = create_autospec(openai.Image)
        mock_image.create_edit = MockImage.create_edit
        with mock.patch("openai.Image", new=mock_image):
            OpenAIInstrumentor().instrument()

            result = openai.Image.create_edit(
                prompt="A cute baby sea otter",
                image="@Otter.png",
                n=2,
            )

            span = self._assert_spans(1)

            name = "openai.image.edit"
            self.assertEqual(span.name, name)
            self.assertEqual(span.attributes[f"{name}.prompt"], "A cute baby sea otter")
            self.assertEqual(span.attributes[f"{name}.image"], "@Otter.png")
            self.assertEqual(span.attributes[f"{name}.n"], 2)
            self.assertEqual(span.attributes[f"{name}.size"], "1024x1024")
            self.assertEqual(span.attributes[f"{name}.response_format"], "url")
            self.assertEqual(
                span.attributes[f"{name}.response.created"], result["created"]
            )
            self.assertEqual(
                span.attributes[f"{name}.response.data.0.url"], result["data"][0]["url"]
            )
            self.assertEqual(
                span.attributes[f"{name}.response.data.1.url"], result["data"][1]["url"]
            )

            self._assert_api_attributes(span)

            OpenAIInstrumentor().uninstrument()

    def test_instrument_image_variation(self):
        mock_image = create_autospec(openai.Image)
        mock_image.create_variation = MockImage.create_variation
        with mock.patch("openai.Image", new=mock_image):
            OpenAIInstrumentor().instrument()

            result = openai.Image.create_variation(
                image="@Otter.png",
                n=2,
            )

            span = self._assert_spans(1)

            name = "openai.image.variation"
            self.assertEqual(span.name, name)
            self.assertEqual(span.attributes[f"{name}.image"], "@Otter.png")
            self.assertEqual(span.attributes[f"{name}.n"], 2)
            self.assertEqual(span.attributes[f"{name}.size"], "1024x1024")
            self.assertEqual(span.attributes[f"{name}.response_format"], "url")
            self.assertEqual(
                span.attributes[f"{name}.response.created"], result["created"]
            )
            self.assertEqual(
                span.attributes[f"{name}.response.data.0.url"], result["data"][0]["url"]
            )
            self.assertEqual(
                span.attributes[f"{name}.response.data.1.url"], result["data"][1]["url"]
            )

            self._assert_api_attributes(span)

            OpenAIInstrumentor().uninstrument()

    def test_instrument_audio_transcribe(self):
        mock_audio = create_autospec(openai.Audio)
        mock_audio.transcribe = MockAudio.transcribe
        with mock.patch("openai.Audio", new=mock_audio):
            OpenAIInstrumentor().instrument()

            result = openai.Audio.transcribe(
                file="audio.mp3",
                model="whisper-1",
            )

            span = self._assert_spans(1)

            name = "openai.audio.transcribe"
            self.assertEqual(span.name, name)
            self.assertEqual(span.attributes[f"{name}.file"], "audio.mp3")
            self.assertEqual(span.attributes[f"{name}.model"], "whisper-1")
            self.assertEqual(span.attributes[f"{name}.prompt"], "")
            self.assertEqual(span.attributes[f"{name}.response_format"], "json")
            self.assertEqual(span.attributes[f"{name}.temperature"], 0.0)
            self.assertEqual(span.attributes[f"{name}.language"], "")
            self.assertEqual(span.attributes[f"{name}.response.text"], result["text"])

            self._assert_api_attributes(span)

            OpenAIInstrumentor().uninstrument()

    def test_instrument_audio_translate(self):
        mock_audio = create_autospec(openai.Audio)
        mock_audio.translate = MockAudio.translate
        with mock.patch("openai.Audio", new=mock_audio):
            OpenAIInstrumentor().instrument()

            result = openai.Audio.translate(
                file="german.m4a",
                model="whisper-1",
            )

            span = self._assert_spans(1)

            name = "openai.audio.translate"
            self.assertEqual(span.name, name)
            self.assertEqual(span.attributes[f"{name}.file"], "german.m4a")
            self.assertEqual(span.attributes[f"{name}.model"], "whisper-1")
            self.assertEqual(span.attributes[f"{name}.prompt"], "")
            self.assertEqual(span.attributes[f"{name}.response_format"], "json")
            self.assertEqual(span.attributes[f"{name}.temperature"], 0.0)
            self.assertEqual(span.attributes[f"{name}.response.text"], result["text"])

            self._assert_api_attributes(span)

            OpenAIInstrumentor().uninstrument()

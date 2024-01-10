import json
import os
import time
from typing import List

import openai
import tiktoken
import weaviate
from dotenv import load_dotenv
from embedia import (
    LLM,
    ChatLLM,
    EmbeddingModel,
    ParamDocumentation,
    TextDoc,
    Tokenizer,
    Tool,
    ToolDocumentation,
    ToolReturn,
    VectorDB,
    VectorDBGetSimilar,
    VectorDBInsert,
)
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from weaviate.embedded import EmbeddedOptions

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class OpenAITokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    async def _tokenize(self, text: str) -> List[int]:
        return tiktoken.encoding_for_model("gpt-3.5-turbo").encode(text)


class OpenAILLM(LLM):
    def __init__(self):
        super().__init__(tokenizer=OpenAITokenizer(), max_input_tokens=4000)

    async def _complete(self, prompt: str) -> str:
        completion = await openai.Completion.acreate(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=500,
            temperature=0.1,
        )
        return completion.choices[0].text


class OpenAIChatLLM(ChatLLM):
    def __init__(self):
        super().__init__(tokenizer=OpenAITokenizer(), max_input_tokens=4096)

    async def _reply(self, prompt: str) -> str:
        completion = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=500,
            messages=[
                {"role": msg.role, "content": msg.content} for msg in self.chat_history
            ],
        )
        return completion.choices[0].message.content


class OpenAIChatLLMCreative(ChatLLM):
    def __init__(self):
        super().__init__(tokenizer=OpenAITokenizer(), max_input_tokens=4096)

    async def _reply(self, prompt: str) -> str:
        completion = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            temperature=1,
            max_tokens=500,
            messages=[
                {"role": msg.role, "content": msg.content} for msg in self.chat_history
            ],
        )
        return completion.choices[0].message.content


class SleepTool(Tool):
    def __init__(self):
        super().__init__(docs={"name": "Sleep Tool", "desc": "Sleeps for 1 second"})

    async def _run(self):
        print("Sleeping for 1 second...")
        time.sleep(1)
        return {"output": "done"}


class PrintTool(Tool):
    def __init__(self):
        super().__init__(
            docs=ToolDocumentation(
                name="Print Tool",
                desc="Prints whatever you want",
                params=[
                    ParamDocumentation(
                        name="text", desc="The text to be printed. Type: String"
                    )
                ],
            )
        )

    async def _run(self, text: str):
        await self.human_confirmation(details={"text": text})
        print(text)
        return ToolReturn(output="done")


class OpenAIEmbedding(EmbeddingModel):
    def __init__(self):
        super().__init__()

    @retry(
        wait=wait_random_exponential(min=1, max=20),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(openai.InvalidRequestError),
    )
    async def _embed(self, input: str):
        result = await openai.Embedding.acreate(
            input=input, model="text-embedding-ada-002"
        )
        return result["data"][0]["embedding"]


class WeaviateDB(VectorDB):
    def __init__(self):
        super().__init__()
        self.client = weaviate.Client(
            embedded_options=EmbeddedOptions(persistence_data_path="./temp/weaviate")
        )
        if not self.client.schema.get()["classes"]:
            self.client.schema.create_class(
                {
                    "class": "Document",
                    "properties": [
                        {
                            "name": "contents",
                            "dataType": ["text"],
                        },
                        {
                            "name": "meta",
                            "dataType": ["text"],
                        },
                    ],
                }
            )

    async def _insert(self, data: VectorDBInsert):
        if not data.meta:
            data.meta = {}
        return self.client.data_object.create(
            data_object={
                "contents": data.text,
                "meta": json.dumps(data.meta),
            },
            class_name="Document",
            uuid=data.id,
            vector=data.embedding,
        )

    async def _get_similar(self, data: VectorDBGetSimilar):
        response = (
            self.client.query.get("Document", ["contents", "meta"])
            .with_near_vector(
                {
                    "vector": data.embedding,
                }
            )
            .with_limit(data.n_results)
            .with_additional(["distance", "id"])
            .do()
        )
        docs = response["data"]["Get"]["Document"]

        result = []
        for doc in docs:
            meta = json.loads(doc["meta"])
            result.append(
                (
                    1 - doc["_additional"]["distance"],
                    TextDoc(
                        id=doc["_additional"]["id"], contents=doc["contents"], meta=meta
                    ),
                )
            )
        return result


class OpenAILLMOptional1(LLM):
    def __init__(self):
        super().__init__(max_input_tokens=4000)

    async def _complete(self, prompt: str) -> str:
        completion = await openai.Completion.acreate(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=500,
            temperature=0.1,
        )
        return completion.choices[0].text


class OpenAILLMOptional2(LLM):
    def __init__(self):
        super().__init__(tokenizer=OpenAITokenizer())

    async def _complete(self, prompt: str) -> str:
        completion = await openai.Completion.acreate(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=500,
            temperature=0.1,
        )
        return completion.choices[0].text


class OpenAILLMOptional3(LLM):
    def __init__(self):
        super().__init__(tokenizer=OpenAITokenizer(), max_input_tokens=2)

    async def _complete(self, prompt: str) -> str:
        completion = await openai.Completion.acreate(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=500,
            temperature=0.1,
        )
        return completion.choices[0].text


class OpenAIChatLLMOptional1(ChatLLM):
    def __init__(self):
        super().__init__(max_input_tokens=4096)

    async def _reply(self, prompt: str) -> str:
        completion = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=500,
            messages=[
                {"role": msg.role, "content": msg.content} for msg in self.chat_history
            ],
        )
        return completion.choices[0].message.content


class OpenAIChatLLMOptional2(ChatLLM):
    def __init__(self):
        super().__init__(tokenizer=OpenAITokenizer())

    async def _reply(self, prompt: str) -> str:
        completion = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=500,
            messages=[
                {"role": msg.role, "content": msg.content} for msg in self.chat_history
            ],
        )
        return completion.choices[0].message.content


class OpenAIChatLLMOptional3(ChatLLM):
    def __init__(self):
        super().__init__(tokenizer=OpenAITokenizer(), max_input_tokens=2)

    async def _reply(self, prompt: str) -> str:
        completion = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=500,
            messages=[
                {"role": msg.role, "content": msg.content} for msg in self.chat_history
            ],
        )
        return completion.choices[0].message.content


class OpenAIChatLLMOptional4(ChatLLM):
    def __init__(self):
        super().__init__(tokenizer=OpenAITokenizer(), max_input_tokens=4096)

    async def _reply(self) -> str:
        completion = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=500,
            messages=[
                {"role": msg.role, "content": msg.content} for msg in self.chat_history
            ],
        )
        return completion.choices[0].message.content


class OpenAIChatLLMBroken(ChatLLM):
    def __init__(self):
        pass

    async def _reply(self) -> str:
        completion = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=500,
            messages=[
                {"role": msg.role, "content": msg.content} for msg in self.chat_history
            ],
        )
        return completion.choices[0].message.content


class PrintToolBroken1(Tool):
    def __init__(self):
        super().__init__(
            docs=ToolDocumentation(
                name="Sleep Tool",
                desc="Sleeps for 1 second",
                params=[
                    {"name": "texts", "desc": "The text to be printed. Type: String"}
                ],
            )
        )

    async def _run(self, text: str):
        await self.human_confirmation(details={"text": text})
        print(text)
        return {"output": "done", "exit_code": 0}


class PrintToolBroken2(Tool):
    def __init__(self):
        super().__init__(
            docs=ToolDocumentation(
                name="Sleep Tool",
                desc="Sleeps for 1 second",
                params=[
                    ParamDocumentation(
                        name="text", desc="The text to be printed. Type: String"
                    )
                ],
            )
        )

    async def _run(self, text: str):
        await self.human_confirmation(details={"text": text})
        print(text)
        return "done"


class PrintToolBroken3(Tool):
    def __init__(self):
        pass

    async def _run(self, text: str):
        await self.human_confirmation(details={"text": text})
        print(text)
        return "done"

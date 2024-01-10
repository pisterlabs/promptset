import base64
from typing import Literal, Optional

from dynamodm import DynaModel  # type: ignore
from openai.types.beta.assistant import Assistant as _Assistant
from openai.types.beta.assistant_create_params import (
    ToolAssistantToolsCode,
    ToolAssistantToolsFunction,
    ToolAssistantToolsRetrieval,
)
from pydantic import BaseModel, Field  # pylint: disable=E0611

from ..tools.storage import StorageBucket
from ._service import Service


class AssistantRequest(BaseModel):
    model: Optional[Literal["gpt-4-1106-preview", "gpt-3.5-turbo-0613"]] = Field(
        default="gpt-4-1106-preview"
    )
    description: Optional[str] = Field(
        default=..., description="Description of the assistant"
    )
    file_ids: Optional[list[str]] = Field(
        default=[], description="File IDs to use for the assistant"
    )
    name: Optional[str] = Field(default=None, description="Name of the assistant")
    instructions: Optional[str] = Field(
        ..., description="Instructions for the assistant"
    )
    tools: Optional[
        list[
            ToolAssistantToolsFunction
            | ToolAssistantToolsCode
            | ToolAssistantToolsRetrieval
        ]
    ] = Field(
        default=[
            ToolAssistantToolsCode(type="code_interpreter"),
            ToolAssistantToolsRetrieval(type="retrieval"),
        ],
        description="The tools to use for the assistant",
    )


class AssistantResponse(_Assistant):
    avatar: str = Field(default=..., description="The avatar of the assistant")


class AssistantModel(DynaModel):
    key: str = Field(..., pk=True)
    sort: str = Field(..., sk=True)
    data: AssistantResponse


class AssistantService(Service[AssistantRequest, AssistantResponse]):
    @property
    def client(self):
        return self.api.beta.assistants

    @property
    def storage(self) -> StorageBucket:
        return StorageBucket()

    async def create_(self, *, key: str, data: AssistantRequest) -> AssistantResponse:
        image_base64, sort = await self._create_image(data)
        avatar = await self.storage.upload(
            body=base64.b64decode(image_base64),
            key=f"{key}/{sort}.png",
            content_type="image/png",
        )
        response = await self.client.create(**data.dict())
        instance = AssistantResponse(**response.dict(), avatar=avatar)
        repository = AssistantModel(key=key, sort=sort, data=instance)
        response = await repository.put()
        return response.data

    async def list_(self, *, key: str):
        response = await AssistantModel.query(pk=key)
        return [item.data for item in response]

    async def get_(self, *, key: str, sort: str):
        response = await AssistantModel.get(pk=key, sk=sort)
        return response.data

    async def update_(self, *, key: str, sort: str, data: AssistantRequest):
        image_base64, sort = await self._create_image(data, sort=sort)
        avatar = await self.storage.upload(
            body=base64.b64decode(image_base64),
            key=f"{key}/{sort}.png",
            content_type="image/png",
        )
        response = await self.client.update(assistant_id=sort, **data.dict())
        instance = AssistantResponse(**response.dict(), avatar=avatar)
        response = await AssistantModel(key=key, sort=sort, data=instance).put()
        return response.data

    async def delete_(self, *, key: str, sort: str):
        await self.client.delete(assistant_id=sort)
        await AssistantModel.delete(pk=key, sk=sort)
        await self.storage.delete(key=f"{key}/{sort}.png")

    async def _create_image(
        self, assistant: AssistantRequest, sort: Optional[str] = None
    ):
        if sort is None:
            response = await self.client.create(**assistant.dict(exclude={"user"}))
            sort = response.id
        else:
            response = await self.client.retrieve(assistant_id=sort)
        model_output = await self.api.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=f"Please generate a succinct yet comprehensive description for creating an avatar for an AI assistant with the following goal: ## Assistant: {response.name}\n{response.instructions}",
        )
        instruction_output = model_output.choices[0].text
        image_base64 = (
            (
                await self.api.images.generate(
                    model="dall-e-3",
                    prompt=instruction_output,
                    size="1024x1024",
                    response_format="b64_json",
                )
            )
            .data[0]
            .b64_json
        )
        assert isinstance(image_base64, str)
        return image_base64, sort

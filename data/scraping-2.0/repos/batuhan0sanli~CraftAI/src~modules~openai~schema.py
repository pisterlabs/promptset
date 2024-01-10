from src.base.base_schema import BaseSchema, SubBaseSchema
from marshmallow import fields
from .enums import OpenAIRole, OpenAIEndpoint


class MessagesSchema(SubBaseSchema):
    role = fields.Enum(OpenAIRole, required=True, )
    content = fields.Str(required=True)


class ModuleSchema(BaseSchema):
    endpoint = fields.Enum(OpenAIEndpoint, load_default=OpenAIEndpoint.chat_completion, required=False)
    messages = fields.List(fields.Nested(MessagesSchema), required=True)

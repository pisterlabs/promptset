from typing import TYPE_CHECKING

from app.testing import register

if TYPE_CHECKING:
    from app.testing.types import FactoryProtocol
    from gpt.models import OpenAiProfile, Reply


@register
def reply(self: "FactoryProtocol", **kwargs: dict) -> "Reply":
    previous_reply = kwargs.pop("previous_reply", None)
    return self.mixer.blend("gpt.Reply", previous_reply=previous_reply, **kwargs)


@register
def openai_profile(self: "FactoryProtocol", **kwargs: dict) -> "OpenAiProfile":
    return self.mixer.blend("gpt.OpenAiProfile", **kwargs)

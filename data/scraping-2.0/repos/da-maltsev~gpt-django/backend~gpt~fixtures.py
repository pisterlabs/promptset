from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from app.testing.factory import FixtureFactory
    from gpt.models import OpenAiProfile, Reply
    from users.models import User


@pytest.fixture
def reply(factory: "FixtureFactory", user: "User") -> "Reply":
    return factory.reply(author=user, previous_reply=None)


@pytest.fixture
def openai_profile(factory: "FixtureFactory") -> "OpenAiProfile":
    return factory.openai_profile(usage_count=0)

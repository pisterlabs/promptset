#  Blackboard-PAGI - LLM Proto-AGI using the Blackboard Pattern
#  Copyright (c) 2023. Andreas Kirsch
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import langchain
import pytest

from blackboard_pagi.testing import fake_llm

langchain.llm_cache = None


def test_fake_llm_query():
    """Test that the fake LLM returns the correct query."""
    llm = fake_llm.FakeLLM(texts={"foobar"})
    assert llm("foo") == "bar"


def test_fake_llm_query_with_stop():
    """Test that the fake LLM returns the correct query."""
    llm = fake_llm.FakeLLM(texts={"foobar"})
    assert llm("foo", stop=["a"]) == "b"


def test_fake_llm_missing_query():
    """Test that the fake LLM raises an error if the query is missing."""
    llm = fake_llm.FakeLLM(texts=set())
    with pytest.raises(NotImplementedError):
        raise ValueError(llm("foo"))

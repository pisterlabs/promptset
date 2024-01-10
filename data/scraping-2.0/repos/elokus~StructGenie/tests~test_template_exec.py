import pytest

from structgenie.driver.openai import OpenAIDriver
from structgenie.engine import StructEngine


@pytest.fixture()
def family_loop_template():
    return """
Generate a Person for each of the following family roles:
{family_roles}

Begin!
Family Members: {family_roles}
---
Family: <list[dict], rule=for each $role in ['father', 'mother', 'son']>
Family.$role: <dict>
Family.$role.name: <str>
Family.$role.age: <int>
"""


@pytest.fixture()
def family_input():
    return {
        "family_roles": ["father", "mother", "son"]
    }


def test_for_loop(family_loop_template, family_input):
    engine = StructEngine.from_template(family_loop_template)
    output, _ = engine.run(family_input, raise_error=True)
    member_roles = [[k for k in member.keys()][0] for member in output["family"]]

    assert all([roles in family_input["family_roles"] for roles in member_roles])
    assert len(member_roles) == len(family_input["family_roles"])


def test_for_loop_openai(family_loop_template, family_input):
    engine = StructEngine.from_template(family_loop_template, driver=OpenAIDriver)
    output, _ = engine.run(family_input, raise_error=True)
    member_roles = [[k for k in member.keys()][0] for member in output["family"]]

    assert all([roles in family_input["family_roles"] for roles in member_roles])
    assert len(member_roles) == len(family_input["family_roles"])


if __name__ == '__main__':
    pytest.main()

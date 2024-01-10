import pytest

from structgenie.driver.openai import OpenAIDriver
from structgenie.driver.openai_vision import OpenAIDriverVision
from structgenie.utils.helper import build_prompt_from_template
from structgenie.utils.parsing import dump_to_yaml_string


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


def test_for_loop_driver_predict(family_loop_template, family_input):
    prompt = build_prompt_from_template(family_loop_template, chat_mode=True)
    prompt = prompt.replace("{family_roles}", str(family_input["family_roles"]))
    inputs = {"input": dump_to_yaml_string({"family_members": family_input})}

    driver = OpenAIDriver.load_driver(prompt)
    output = driver.predict(**inputs)
    assert isinstance(output, str)


def test_for_loop_driver_metrics(family_loop_template, family_input):
    prompt = build_prompt_from_template(family_loop_template, chat_mode=True)
    prompt = prompt.replace("{family_roles}", str(family_input["family_roles"]))
    inputs = {"input": dump_to_yaml_string({"family_members": family_input})}

    driver = OpenAIDriver.load_driver(prompt)
    output, metrics = driver.predict_and_measure(**inputs)
    print(metrics)
    assert isinstance(output, str)
    assert isinstance(metrics, dict)

def test_vision_driver():
    template = """Create a list of all groceries found in the images below
    
    Begin!
    _image: <image>
    ---
    Grocery List: <list[str>
    """
    prompt = build_prompt_from_template(template, chat_mode=True)
    inputs = {"input": "", "image_path": "https://media.moemax.com/i/moemax/PIhZbgg1LUzKWNJJjWVdN22A/?fmt=auto&%24hq%24=&w=1200"}

    driver = OpenAIDriverVision.load_driver(prompt, llm_kwargs={"max_tokens": 1000})
    output = driver.predict(**inputs)
    print(output)
    assert isinstance(output, str)

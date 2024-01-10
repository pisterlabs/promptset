from abcli.options import Options
from openai_cli.vision.completion import complete_object


def test_vision_complete():
    assert complete_object(
        object_name="2023-11-12-12-03-40-85851",
        options=Options("Arbutus16"),
        prompt="describe these images",
    )

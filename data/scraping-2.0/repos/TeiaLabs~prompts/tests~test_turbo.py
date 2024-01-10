from prompts import (
    DynamicPrompt,
    OpenAIModelSettings,
    PromptRole,
    TemplateInputs,
    TurboPrompt,
)


def test_turbo_all_none():
    tp = TurboPrompt()
    assert len(tp.prompts) == 0
    tp.add_system_message(message="You are an AI system that fixes text")
    tp.add_user_message(message="fix this text: she no went to the store")
    tp.add_assistant_message(message="fixed text: she did not go to the store")
    tp.add_user_message(message="fix this text: he is no smart")

    assert len(tp.prompts) == 4

    text = tp.build()
    expected = [
        {
            "role": "system",
            "content": "You are an AI system that fixes text",
        },
        {
            "role": "user",
            "content": "fix this text: she no went to the store",
        },
        {
            "role": "assistant",
            "content": "fixed text: she did not go to the store",
        },
        {"role": "user", "content": "fix this text: he is no smart"},
    ]

    assert text == expected


def test_turbo():
    system = DynamicPrompt("<message>")
    user = DynamicPrompt("<user_name>: <message>")
    assistant = DynamicPrompt("answer: <message>")

    tp = TurboPrompt(
        system_templates=system,
        user_templates=user,
        assistant_templates=assistant,
    )
    assert len(tp.prompts) == 0
    tp.add_system_message(message="You are a chatbot")
    tp.add_user_message(user_name="Qui-gon", message="may the force")
    tp.add_assistant_message(message="be with you")
    assert len(tp.prompts) == 3
    text = tp.build()
    expected = [
        {"role": "system", "content": "You are a chatbot"},
        {"role": "user", "content": "Qui-gon: may the force"},
        {"role": "assistant", "content": "answer: be with you"},
    ]
    assert text == expected


def test_from_file():
    tp = TurboPrompt.from_file("samples/turbo.prompt.yaml")

    tp.add_system_message()
    tp.add_user_message(user_name="Qui-gon", message="Hey!")
    tp.add_assistant_message(message="Hello Jonatas! How can I help you today?")

    text = tp.build()
    expected = [
        {"role": "system", "content": "You are a chatbot\n"},
        {"role": "user", "content": "Qui-gon: Hey!\n"},
        {
            "role": "assistant",
            "content": "answer: Hello Jonatas! How can I help you today?\n",
        },
    ]
    assert text == expected

    assert tp.name == "basic_turbo_prompt"
    assert tp.description == "Basic turbo prompt example"
    from prompts.schemas import OpenAIModelSettings

    assert tp.settings == OpenAIModelSettings(
        **{
            "temperature": 0.15,
            "model": "gpt-3.5-turbo",
            "max_tokens": 32,
        }
    )


def test_from_file_with_initial_template_data():
    tp = TurboPrompt.from_file("samples/sample.past.yaml")

    # Check the content of the past messages
    assert (
        tp.prompts[0]["content"]
        == "You are an AI that fixes code issues:\nLanguage: python\n"
    )
    assert (
        tp.prompts[1]["content"]
        == "Code to check:\n```\ndef sum_numbers(a, b):\n    return a * b\n\n```\n"
    )
    assert (
        tp.prompts[2]["content"]
        == "Bug Description:\nThe function should return the sum of a and b, not their product.\n\n"
    )

    # Add a new user message with a different code
    source_code = "def multiply_numbers(a, b):\n    return a - b\n"
    tp.add_user_message(source_code=source_code)

    # Check the built prompts
    expected_messages = [
        "You are an AI that fixes code issues:\nLanguage: python\n",
        "Code to check:\n```\ndef sum_numbers(a, b):\n    return a * b\n\n```\n",
        "Bug Description:\nThe function should return the sum of a and b, not their product.\n\n",
        "Code to check:\n```\ndef multiply_numbers(a, b):\n    return a - b\n\n```\n",
    ]
    text = tp.build()
    assert [msg["content"] for msg in text] == expected_messages

    # Check the title and settings attributes
    assert tp.name == "turbo_prompt_with_examples"
    assert (
        tp.description
        == "Example of turbo prompt with initial_template_data (few-shot)"
    )
    sets = tp.settings
    assert sets is not None
    assert sets.model == "gpt-3.5-turbo"


def test_a_from_settings():
    tp = TurboPrompt.from_file("samples/complex.yaml")
    built = tp.build()
    assert isinstance(built, list)
    assert isinstance(built[0], dict)
    assert len(built) == 3
    assert built[-1]["role"] == "assistant"

    tp = TurboPrompt.from_settings(
        name="turbo_prompt_inline",
        description="",
        system_template="<message>",
        user_template="Q:<message>",
        assistant_template="A:",
        settings=OpenAIModelSettings(model="gpt-4"),
        initial_template_data=[
            TemplateInputs(
                inputs={"message": "You are an AI."}, role=PromptRole.SYSTEM
            )
        ],
    )

    content = tp.build()
    assert len(content) == 1
    print(content)
    assert content[0] == {
        "role": "system",
        "content": "You are an AI.",
    }

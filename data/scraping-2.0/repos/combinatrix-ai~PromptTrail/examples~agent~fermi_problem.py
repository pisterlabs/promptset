# In this example, we create a bot that can solve Fermi Problem using PromptTrail.
# Fermi Problem is a kind of problem that requires you to estimate the quantitative answer to a question using common sense and logic.
# For example, "How many cats in Japan?" is a Fermi Problem, as shown below. (This example is actually from OpenAI API)
# We use OpenAI API to generate text, and use Python to calculate the answer.
# Let's see how it works!

import logging
import os

from prompttrail.agent.hooks import (
    BooleanHook,
    EvaluatePythonCodeHook,
    ExtractMarkdownCodeBlockHook,
    ResetDataHook,
)
from prompttrail.agent.templates import (
    BreakTemplate,
    IfTemplate,
    LoopTemplate,
    MessageTemplate,
    UserInputTextTemplate,
)
from prompttrail.agent.templates.openai import (
    OpenAIGenerateTemplate as GenerateTemplate,
)
from prompttrail.agent.user_interaction import (
    OneTurnConversationUserInteractionTextMockProvider,
    UserInteractionTextCLIProvider,
)
from prompttrail.core import Message
from prompttrail.core.mocks import OneTurnConversationMockProvider
from prompttrail.core.utils import is_in_test_env

logging.basicConfig(level=logging.INFO)

agent_template = LoopTemplate(
    [
        MessageTemplate(
            # First, let's give an instruction to the API
            # In OpenAI API, system is a special role that gives instruction to the API
            template_id="instruction",
            role="system",
            content="""
You're a helpful assistant to solve Fermi Problem.
Answer the equation to estimate the answer to the user's query.

For example, if user ask you `How many cats in Japan?`, your answer should look like below.
Note that you must return python expression to for user to calculate it using Python.

Thoughts:
- I don't have no knowledge about the number of cats in Japan. However, I can estimate it using available knowledge.
- As of 2019, about 67% of households in the United States owned either dogs or cats, with about 49% of households owning cats (source: American Pet Products Association).
- Assuming the trend in Japan follows a similar pattern to the U.S., we can look at the total number of households in Japan, and estimate that about 49% of them own a cat.
- The total number of households in Japan is about 53 million (source: Japan's Ministry of Internal Affairs and Communications, 2020).
- We also need to consider the average number of cats per household.
- In the U.S, the average number of cats per household is 2.1 (source: American Veterinary Medical Association).
- If we apply this to Japan, we can estimate the total number of cats in Japan.
- Now I can estimate the answer.

Equation to be calculated:
- Total Number of Cats in Japan = Total Number of Households in Japan * Rate of Households owning cats  * Average Number of Cats Per Household

Calculation:
```python
5300000 * 0.49 * 2.1
```""",
        ),
        LoopTemplate(
            template_id="fermi_problem_loop",
            templates=[
                # Then, we can start the conversation with user
                # First, we ask user for their question
                # UserInputTextTemplate is a template that ask user for their input.
                # As we see later, we use CLIRunner to run this model in CLI, so the input is given via console.
                first := UserInputTextTemplate(
                    template_id="ask_question",
                    # Note: we can refer to this template later, so we give it a name: "first" with walrus operator.
                    # You can also use template_id to refer to a template.
                    role="user",
                    description="Input:",
                    default="How many elephants in Japan?",
                ),
                GenerateTemplate(
                    template_id="generate_answer",
                    role="assistant",
                    # Now we have the user's question, we can ask the API to generate the answer
                    # Let's use GenerateTemplate to do this
                    # GenerateTemplate make the content of the message using the model.
                    # Previous message is used as context, so the model can generate the answer based on the question.
                    after_transform=[
                        # This is where things get interesting!
                        # You can extract code block from markdown using ExtractMarkdownCodeBlockHook from generated content
                        # As we give an example, the API may include something like this in their response, which is stored in this message
                        #  ```python
                        #     5300000 * 0.49 * 2.1
                        #  ```
                        # This hook will extract a Python code block and store it in key: "python_segment"
                        ExtractMarkdownCodeBlockHook(
                            key="python_segment", lang="python"
                        ),
                        # Then, you can evaluate the code block using EvaluatePythonCodeHook
                        # Yeah, this is a bit dangerous, we will provide safer way to do this in the future.
                        # Note that the key "python_segment" from ExtractMarkdownCodeBlockHook is used here
                        # The result of the evaluation is stored in key: "answer"
                        EvaluatePythonCodeHook(key="answer", code="python_segment"),
                    ],
                ),
                IfTemplate(
                    true_template=MessageTemplate(
                        role="assistant",
                        content="LLM seems to unable to estimate. Try different question! Starting over...",
                    ),
                    false_template=BreakTemplate(),
                    condition=BooleanHook(lambda state: "answer" not in state.data),
                ),
            ],
            before_transform=[ResetDataHook()],
        ),
        MessageTemplate(
            # You can also give assistant message without using model, as if the assistant said it
            # In this case, we want to ask user if the answer is satisfied or not
            # Analysing the user response is always hard, so we let the API to decide
            # First, we must ask user for their feedback
            # Let's ask user for question!
            template_id="gather_feedback",
            role="assistant",
            content="The answer is {{ answer }} . Satisfied?",
        ),
        UserInputTextTemplate(
            # Here is where we ask user for their feedback
            template_id="get_feedback",
            role="user",
            description="Input:",
            default="Yes, I'm satisfied.",
        ),
        MessageTemplate(
            # Based on the feedback, we can decide to retry or end the conversation
            # Ask the API to analyze the user's sentiment
            template_id="instruction_sentiment",
            role="assistant",
            content="The user has stated their feedback. If you think the user is satisified, you must answer `END`. Otherwise, you must answer `RETRY`.",
        ),
        check_end := GenerateTemplate(
            template_id="analyze_sentiment",
            role="assistant",
            # API will return END or RETRY (mostly!)
        ),
    ],
    # Then, we can decide to end the conversation or retry.
    # We use LoopTemplate, so if we don't exit the conversation, we will go to top of loop.
    # Check if the loop is finished, see exit_condition below.
    exit_condition=BooleanHook(lambda state: state.get_last_message().content == "END"),
)

# Then, let's run this agent!
# You can run templates using runner.
# This runner runs models in cli.
from prompttrail.agent.runners import CommandLineRunner  # noqa: E402

# Import some classes to interact with OpenAI API
# You can just use these classes if you directly use OpenAI API. See examples/model/openai.py for more details.
from prompttrail.models.openai import (  # noqa: E402
    OpenAIChatCompletionModel,
    OpenAIModelConfiguration,
    OpenAIModelParameters,
)

# We will provide other runner, which will enable you to input/output via HTTP, etc... in the future.

# It's a little bit off-topic, but we can mock the API to respond in automatically for testing!
# Actually, we're using this example script in the test! (See tests/provider/test_openai.py)
# See the last part of this file for more details.

if not is_in_test_env():
    # First, let's see how the agent works in CLI (without mocking)!
    # Just set up the runner and run it!
    runner = CommandLineRunner(
        model=OpenAIChatCompletionModel(
            configuration=OpenAIModelConfiguration(
                api_key=os.environ.get("OPENAI_API_KEY", "")
            )
        ),
        parameters=OpenAIModelParameters(model_name="gpt-4"),
        template=agent_template,
        user_interaction_provider=UserInteractionTextCLIProvider(),
    )
    if __name__ == "__main__":
        conversation = runner.run()
        # You can keep the conversation data for later use!
        print("=== Summary ===")
        print(conversation)
else:
    # Here, we will run the agent in automatically for testing!
    # If you want to see how the automatic agent works, you can run the agent manually with setting environment variable CI=true or DEBUG=true!
    runner = CommandLineRunner(
        # Use mock model in CI or DEBUG
        model=OpenAIChatCompletionModel(
            configuration=OpenAIModelConfiguration(
                # Of course, same arguments as OpenAIChatCompletionModel can be used
                api_key=os.environ.get("OPENAI_API_KEY", ""),
                # You can define the behaviour of the mock model using mock_provider
                mock_provider=OneTurnConversationMockProvider(
                    conversation_table={
                        "How many cats in Japan?": Message(
                            content="""Thoughts: ...
        Calculation:
        ```python
        5300000 * 0.49 * 2.1
        ```
        """,
                            sender="assistant",
                        ),
                        "The user has stated their feedback. If you think the user is satisified, you must answer `END`. Otherwise, you must answer `RETRY`.": Message(
                            content="END", sender="assistant"
                        ),
                    },
                    sender="assistant",
                ),
            ),
        ),
        user_interaction_provider=OneTurnConversationUserInteractionTextMockProvider(
            conversation_table={
                # 5300000 * 0.49 * 2.1 = 5453700.0
                agent_template.templates[0].content: "How many cats in Japan?",  # type: ignore
                "The answer is 5453700.0 . Satisfied?": "OK",
            }
        ),
        parameters=OpenAIModelParameters(model_name="gpt-4"),
        template=agent_template,
    )
    if __name__ == "__main__":
        conversation = runner.run()
        print("=== Summary ===")
        print(conversation)

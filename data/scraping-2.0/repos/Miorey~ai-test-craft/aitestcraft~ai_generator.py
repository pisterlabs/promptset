from openai import OpenAI
from aitestcraft.conf_representation import ProjectConfig, Overwrite
from halo import Halo
from pathlib import Path


def execute_test_cover(gen_setup: ProjectConfig, open_ai_client: OpenAI):
    language = gen_setup.language
    language_version = str(gen_setup.language_version or "")
    conversation_history = [
        {
            "role": "system",
            "content": f"You are a f{language} {language_version} developer"
        }]

    spinner = Halo(text="Loading...", spinner='dots')
    for to_cover in gen_setup.files:
        spinner.start(f"{to_cover.code} => {to_cover.test}")

        # AI-TEST: Mock the open usage to avoid FileNotFound exception
        with open(to_cover.code, 'r') as file:
            content = file.read()
        additional_comments = gen_setup.additional_comments if gen_setup.additional_comments else []

        conversation_history.append(
            {
                "role": "user",
                "content": f"""
                            Write the unit test of the following code. 
                            The test should follow those rules:
                             - THE TEST SHOULD COVER 100% of the code.
                             - In the imports take in account that the test is in {to_cover.code} and the test in {to_cover.test}.
                             - The comments starting with `AI-TEST:` take them in consideration. 
                             - {', '.join(additional_comments)}
                             - BE SURE YOU USE ONLY THAT YOU WELL `import` the requirements.
                             - The test should be simple and with a cyclomatic complexity as lower as possible.

                            Your answer should contain:
                             - NO SYNTAX HIGHLIGHTING.
                             - no introduction or explanation.
                             - ALL the test should be in the same snippet.
                             - Avoid to give in the answer the code given in the demand
                            ```
                                {content}
                            ```
                            """
            }
        )

        if Path(to_cover.test).is_file() and gen_setup.overwrite == Overwrite.NEVER:
            spinner.warn()
            with open(to_cover.test, 'r') as file:
                content = file.read()
                conversation_history.append({'role': 'assistant', 'content': content})
            continue

        # AI-TEST: mock the following openai call
        completion = open_ai_client.chat.completions.create(
            model=gen_setup.model,
            messages=conversation_history
        )
        ai_answer: str = completion.choices[0].message.content
        conversation_history.append({'role': 'assistant', 'content': ai_answer})
        only_code = ai_answer.replace(f"```{language}", "")
        only_code = only_code.replace("```", "")
        with open(to_cover.test, 'w') as f:
            f.write(only_code)
        spinner.succeed()

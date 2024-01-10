from reflexion.prompts import PROMPTS_DIR

from reflexion.environments.programming import InternalTestingEnv
from reflexion.environments.programming import PythonTestingEnv
from reflexion.llms import OpenAIChatLLM, MockLLM

import dotenv

dotenv.load_dotenv()

import logging

logging.basicConfig(level=logging.INFO)

function_signature = "reverse_words(s: str) -> str:"
docstring = "Input to this function is a string containing multiple words. Your goal is to reverse the order of the words."

solution = """
def reverse_words(s: str) -> str:
    return s[::-1]
"""

tests = """
Tests:
assert reverse_words("Hello World") == "olleH dlroW"  # This test will fail.
assert reverse_words("This is a test") == "sihT si a tset"
assert reverse_words("abc") == "cba"
"""

llm = MockLLM(responses =[tests])

local_env = PythonTestingEnv()
internal_env = InternalTestingEnv(
    function_signature=function_signature, docstring=docstring, language="python", local_env=local_env, llm=llm
)

rewards, messages = internal_env.step(solution)

def test_step():
    print(internal_env.generated_tests)
    assert tuple(rewards) == (False, False, True)
    assert messages == ("AssertionError, actual left-hand value was: dlroW olleH", "AssertionError, actual left-hand value was: tset a si sihT", "passed")

def test_format_feedback():
    feedback = internal_env.format_feedback(rewards, messages)
    print(feedback)
    assert internal_env.format_feedback(rewards, messages).strip().strip('\n') == """
Tests Passed:
assert reverse_words("abc") == "cba"

Tests Failed:
assert reverse_words("Hello World") == "olleH dlroW"  # This test will fail.
Message: AssertionError, actual left-hand value was: dlroW olleH

assert reverse_words("This is a test") == "sihT si a tset"
Message: AssertionError, actual left-hand value was: tset a si sihT
    """.strip().strip('\n')


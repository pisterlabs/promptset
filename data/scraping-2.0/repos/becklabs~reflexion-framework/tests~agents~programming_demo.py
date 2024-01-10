import os

from reflexion.agents.programming import ProgrammingReflexionAgent
from reflexion.prompts import PROMPTS_DIR
from reflexion.actors import LanguageFunction

from reflexion.environments.programming import InternalTestingEnv
from reflexion.environments.programming import PythonTestingEnv
from reflexion.llms import OpenAIChatLLM, MockLLM
from langchain.callbacks import get_openai_callback

import logging
logging.basicConfig(level=logging.INFO)

function_signature = "def separate_paren_groups(paren_string: str) -> List[str]:"

docstring = """
Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
separate those group into separate strings and return the list of those.
Separate groups are balanced (each open brace is properly closed) and not nested within each other
Ignore any spaces in the input string.
>>> separate_paren_groups('( ) (( )) (( )( ))')
['()', '(())', '(()())']
"""

llm = OpenAIChatLLM(model_name="gpt-4", temperature=0, max_tokens=1000)

# PROMPTS
simple_implementation_function = LanguageFunction.from_yaml(
    os.path.join(
        PROMPTS_DIR,
        "programming",
        "v1",
        "implementation",
        "python",
        "simple.yaml",
    ),
    llm=llm,
)

implementation_function = LanguageFunction.from_yaml(
    os.path.join(
        PROMPTS_DIR,
        "programming",
        "v1",
        "implementation",
        "python",
        "reflexion.yaml",
    ),
    llm=llm,
)

reflexion_function = LanguageFunction.from_yaml(
    os.path.join(
        PROMPTS_DIR, "programming", "v1", "reflecting", "python", "reflexion.yaml"
    ),
    llm=llm,
)

local_env = PythonTestingEnv(timeout=10)

agent = ProgrammingReflexionAgent(
    function_signature=function_signature,
    docstring=docstring,
    testing_env=InternalTestingEnv(
        function_signature=function_signature,
        docstring=docstring,
         language="python", local_env=local_env, llm=llm
    ),
    simple_implemenation_function=simple_implementation_function,
    implementation_function=implementation_function,
    reflection_function=reflexion_function,
)

with get_openai_callback() as cb:
    reward, message = agent.step()
    print(reward)
    print(message)
print("Total cost", cb.total_cost)

with get_openai_callback() as cb:
    reward, message = agent.step()
    print(reward)
    print(message)
print("Total cost", cb.total_cost)

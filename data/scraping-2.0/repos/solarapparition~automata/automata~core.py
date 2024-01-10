"""Core automata functionality."""

from functools import lru_cache, partial
from pathlib import Path
from typing import Callable, Dict, List, Union

from langchain import LLMChain, PromptTemplate
from langchain.agents import Tool
import yaml

from automata.engines import create_engine
from automata.builtin_functions import load_builtin_function
from automata.validation import (
    load_input_validator,
    load_output_validator,
    IOValidator,
)
from automata.reasoning import (
    AutomatonAgent,
    AutomatonExecutor,
    AutomatonOutputParser,
)
from automata.knowledge import load_knowledge
from automata.loaders import (
    get_full_name,
    get_role_info,
    load_automaton_data,
)
from automata.planners import load_planner
from automata.reflection import load_reflect
from automata.sessions import add_session_handling
from automata.types import Automaton, AutomatonRunner
from automata.utilities import generate_timestamp_id
from .utilities.importing import quick_import


def create_automaton_prompt(
    objective: str,
    self_instructions: List[str],
    self_imperatives: List[str],
    role_info: Dict[str, str],
    sub_automata: List[Tool],
    requester_full_name: str,
    background_knowledge: Union[str, None],
) -> PromptTemplate:
    """Put together a prompt for an automaton."""

    imperatives = role_info["imperatives"] + (self_imperatives or [])
    imperatives = "\n".join([f"- {imperative}" for imperative in imperatives]) or "N/A"

    instructions = (self_instructions or []) + role_info["instructions"]
    instructions = (
        "\n".join([f"- {instruction}" for instruction in instructions]) or "N/A"
    )
    affixes: Dict[str, str] = {
        key: val.strip()
        for key, val in yaml.load(
            Path("automata/prompts/automaton.yml").read_text(encoding="utf-8"),
            Loader=yaml.FullLoader,
        ).items()
    }

    prefix = affixes["prefix"].format(
        role_description=role_info["description"],
        imperatives=imperatives,
        background_knowledge=background_knowledge,
    )

    suffix = (
        affixes["suffix"]
        .replace("{instructions}", instructions)
        .replace("{objective}", objective)
        .replace("{requester}", requester_full_name)
    )
    prompt = AutomatonAgent.create_prompt(
        sub_automata,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "agent_scratchpad"],
        format_instructions=role_info["output_format"],
    )
    return prompt


@lru_cache(maxsize=None)
def load_automaton(
    automaton_id: str,
    automata_location: Path,
    requester_session_id: str,
    requester_id: str,
) -> Automaton:
    """Load an automaton from a YAML file."""

    data = load_automaton_data(automata_location / automaton_id)
    automaton_path = automata_location / automaton_id
    full_name = f"{data['name']} ({data['role']} {data['rank']})"
    engine = create_engine(data["engine"])

    input_requirements = data["input_requirements"]
    input_requirements_prompt = (
        "\n".join([f"- {req}" for req in input_requirements])
        if input_requirements
        else "None"
    )
    description_and_input = (
        data["description"] + f" Input requirements:\n{input_requirements_prompt}"
    )

    input_validator = load_input_validator(
        data["input_validator"], input_requirements, automaton_id, automata_location
    )

    def run_builtin_function(*args, **kwargs) -> str:
        run = load_builtin_function(
            automaton_id,
            automata_location,
            data,
            engine,
            requester_id=requester_id,
        )
        return run(*args, **kwargs)

    self_session_id = generate_timestamp_id()

    # lazy load sub-automata until needed
    def run_core_automaton(*args, **kwargs) -> str:
        request = args[0]
        output_validator: Union[IOValidator, None] = load_output_validator(
            data["output_validator"], request=request, file_name=automaton_id
        )
        reflect: Union[Callable, None] = load_reflect(
            automata_location / automaton_id, data["reflect"]
        )
        planner = load_planner(automaton_path, data["planner"])
        sub_automata = [
            load_automaton(
                sub_automata_id,
                requester_session_id=self_session_id,
                requester_id=automaton_id,
                automata_location=automata_location,
            )
            for sub_automata_id in data["sub_automata"]
        ]
        create_background_knowledge = load_knowledge(
            automaton_path,
            data["knowledge"],
        )
        background_knowledge = (
            create_background_knowledge(args[0])
            if create_background_knowledge
            else None
        )
        prompt = create_automaton_prompt(
            objective=data["objective"],
            self_instructions=data["instructions"],
            self_imperatives=data["imperatives"],
            role_info=get_role_info(data["role"]),
            background_knowledge=background_knowledge,
            sub_automata=sub_automata,
            requester_full_name=get_full_name(requester_id, automata_location),
        )
        # print(prompt.format(input="blah", agent_scratchpad={}))
        # breakpoint()
        agent_executor = AutomatonExecutor.from_agent_and_tools(
            agent=AutomatonAgent(
                llm_chain=LLMChain(llm=engine, prompt=prompt),
                allowed_tools=[sub_automaton.name for sub_automaton in sub_automata],
                output_parser=AutomatonOutputParser(validate_output=output_validator),
                reflect=reflect,
                planner=planner,
            ),
            tools=sub_automata,
            verbose=True,
            max_iterations=None,
            max_execution_time=None,
        )
        return agent_executor.run(*args, **kwargs)

    runner_name: str = data["runner"]

    if runner_name.endswith(".py"):
        custom_runner: AutomatonRunner = quick_import(
            automata_location / runner_name
        ).run
        runner = partial(
            custom_runner,
            automaton_id=automaton_id,
            automata_data=data,
            requester_id=requester_id,
        )
    elif runner_name == "default_function_runner":
        runner = run_builtin_function
    elif runner_name == "default_automaton_runner":
        runner = run_core_automaton
    else:
        raise NotImplementedError(f"Unknown runner {runner_name}")

    automaton = Tool(
        full_name,
        add_session_handling(
            runner,
            automaton_id=automaton_id,
            automata_location=automata_location,
            session_id=self_session_id,
            full_name=full_name,
            requester_id=requester_id,
            input_validator=input_validator,
            requester_session_id=requester_session_id,
        ),
        description_and_input,
    )
    return automaton

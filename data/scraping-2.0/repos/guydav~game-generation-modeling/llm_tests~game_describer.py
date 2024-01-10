import argparse
from itertools import groupby
import os
import sys
import typing

import logging
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)

import backoff
import inflect
import openai

import tabulate
import tatsu, tatsu.ast, tatsu.grammars
from tqdm import tqdm

# For some reason utils needs to be imported first?
sys.path.append(os.path.abspath('./reward-machine'))
sys.path.append(os.path.abspath('../reward-machine'))
from utils import OBJECTS_BY_ROOM_AND_TYPE, extract_predicate_function_name, extract_variables, extract_variable_type_mapping
from describer_lm_prompts import compile_prompts_from_data
from preference_handler import PredicateType

from ast_parser import SETUP, PREFERENCES, TERMINAL, SCORING
from ast_printer import ast_section_to_string
from ast_utils import cached_load_and_parse_games_from_file

DEFAULT_GRAMMAR_PATH = "../dsl/dsl.ebnf"
DEFAULT_GAMES_PATH = "../dsl/interactive-beta.pddl"

PREDICATE_DESCRIPTIONS = {
    "above": "{0} is above {1}",
    "adjacent": "{0} is adjacent to {1}",
    "adjacent_side_3_args": "{2} is adjacent to the {1} of {0}",
    "adjacent_side_4_args": "the {1} of {0} is adjacent to the {3} of {2}",
    "agent_crouches": "the agent is crouching",
    "agent_holds": "the agent is holding {0}",
    "between": "{1} is between {0} and {2}",
    "broken": "{0} is broken",
    "equal_x_position": "{0} and {1} have the same x position",
    "equal_y_position": "{0} and {1} have the same y position",
    "equal_z_position": "{0} and {1} have the same z position",
    "faces": "{0} is facing {1}",
    "game_over": "it is the last state in the game",
    "game_start": "it is the first state in the game",
    "in": "{1} is inside of {0}",
    "in_motion": "{0} is in motion",
    "is_setup_object": "{0} is used in the setup",
    "near": "{0} is near {1}",
    "object_orientation": "{0} is oriented {1}",
    "on": "{1} is on {0}",
    "open": "{0} is open",
    "opposite": "{0} is opposite {1}",
    "rug_color_under": "the color of the rug under {0} is {1}",
    "same_color": "{0} is the same color as {1}",
    "same_object": "{0} is the same object as {1}",
    "same_type": "{0} is of the same type as {1}",
    "touch": "{0} touches {1}",
    "toggled_on": "{0} is toggled on",
}

FUNCTION_DESCRIPTIONS = {
    "building_size": "the number of obects in building {0}",
    "distance": "the distance between {0} and {1}",
    "distance_side": "the distance between {2} and the {1} of {0}",
    "x_position": "the x position of {0}",
    "y_position": "the y position of {0}",
    "z_position": "the z position of {0}",
}

SETUP_HEADER = "In order to set up the game, "
PREFERENCES_HEADER = "\nThe preferences of the game are:"
TERMINAL_HEADER = "\nThe game ends when "
SCORING_HEADER = "\nAt the end of the game, the player's score is "

STYLE_HTML = """
<style>
    .table td, .table th {
        min-width: 40em;
        max-width: 60em;
    }
    pre {
        white-space: pre-wrap;
        max-height: 60em;
        overflow: auto;
        display: inline-block;
    }
</style>
"""

TABLE_HTML_TEMPLATE = """
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>Game DSL to natural language translations </title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    {0}
</head>
<body>
    <div>
        {1}
    </div>
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/popper.js@1.12.9/dist/umd/popper.min.js" integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
</body>
"""

class GameDescriber():
    def __init__(self,
                 grammar_path: str = DEFAULT_GRAMMAR_PATH,
                 openai_model_str: typing.Optional[str] = None,
                 max_openai_tokens: int = 512,
                 openai_temperature: float = 0.0):

        grammar = open(grammar_path).read()
        self.grammar_parser = typing.cast(tatsu.grammars.Grammar, tatsu.compile(grammar))
        self.engine = inflect.engine()

        self.preference_index =  None
        self.external_forall_preference_mappings = None

        if openai_model_str is not None:
            self.openai_model_str = openai_model_str
            openai_key = os.environ.get("OPENAI_TOKEN")
            if openai_key is None:
                openai_key = os.environ.get("OPENAI_API_KEY")
                if openai_key is None:
                    raise ValueError("Error: OPENAI_TOKEN/OPENAI_API_KEY environment variable is not set")

        self.openai_client = openai.OpenAI(api_key=openai_key)
        self.max_openai_tokens = max_openai_tokens
        self.openai_temperature = openai_temperature

    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    def _query_openai(self, prompt: str):
        '''
        Query the specified openai model with the given prompt, and return the response. Assumes
        that the API key has already been set. Retries with exponentially-increasing delays in
        case of rate limit errors
        '''

        response = self.openai_client.chat.completions.create(
            model=self.openai_model_str,
            max_tokens=self.max_openai_tokens,
            temperature=self.openai_temperature,
            messages=[{"role": "user", "content": prompt}]
        )

        reponse_content = response.choices[0].message.content

        return reponse_content

    def _indent(self, description: str, num_spaces: int = 4):
        '''
        Add a specified number of spaces to each line passed in
        '''
        lines = description.split("\n")
        return "\n".join([f"{' ' * num_spaces}{line}" if line !="" else line for line in lines])

    def _extract_game_info(self, ast: typing.Union[list, tuple, tatsu.ast.AST], info_dict: typing.Dict):
        '''
        Recursively extract the game's name, domain, setup, preferences, terminal conditions, and
        scoring (if they exist)
        '''
        if isinstance(ast, tuple) or isinstance(ast, list):
            for item in ast:
                self._extract_game_info(item, info_dict)

        elif isinstance(ast, tatsu.ast.AST):
            rule = ast["parseinfo"].rule  # type: ignore
            if rule == "game_def":
                info_dict["game_name"] = typing.cast(str, ast["game_name"])

            elif rule == "domain_def":
                info_dict["domain_name"] = typing.cast(str, ast["domain_name"]).split("-")[0]

            elif rule == "setup":
                info_dict["setup"] = ast["setup"]

            elif rule == "preferences":
                # Handle games with single preference
                if isinstance(ast["preferences"], tatsu.ast.AST):
                    info_dict["preferences"] = [ast["preferences"]]
                else:
                    info_dict["preferences"] = [ast["preferences"]]

            elif rule == "terminal":
                info_dict["terminal"] = ast["terminal"]

            elif rule == "scoring_expr":
                info_dict["scoring"] = ast

    def _predicate_type(self, predicate: tatsu.ast.AST) -> PredicateType:
        '''
        Returns the temporal logic type of a given predicate
        '''
        if "once_pred" in predicate.keys():
            return PredicateType.ONCE

        elif "once_measure_pred" in predicate.keys():
            return PredicateType.ONCE_MEASURE

        elif "hold_pred" in predicate.keys():

            if "while_preds" in predicate.keys():
                return PredicateType.HOLD_WHILE

            return PredicateType.HOLD

        else:
            raise ValueError(f"Error: predicate does not have a temporal logic type: {predicate.keys()}")

    def _extract_name_and_types(self, scoring_expression: tatsu.ast.AST) -> typing.Tuple[str, typing.Optional[typing.Sequence[str]]]:
        '''
        Helper function to extract the name of the preference being scored, as well as any of the object types that have been
        passed to it using the ":" syntax
        '''

        if isinstance(scoring_expression, tatsu.ast.AST):

            if "name_and_types" in scoring_expression.keys():

                name_and_types = typing.cast(tatsu.ast.AST, scoring_expression["name_and_types"])
                preference_name = name_and_types["pref_name"]

                if isinstance(name_and_types["object_types"], tatsu.ast.AST):
                    object_types = [name_and_types["object_types"]["type_name"]]  # type: ignore

                elif isinstance(name_and_types["object_types"], list):
                    object_types = [object_type["type_name"] for object_type in name_and_types["object_types"]]  # type: ignore

                else:
                    object_types = None

                if object_types is not None:
                    object_types = [type_name["terminal"] for type_name in object_types]

                return str(preference_name), object_types

            else:
                for key in scoring_expression.keys():
                    if key != "parseinfo":
                        preference_name, object_types = self._extract_name_and_types(scoring_expression[key])

                        if preference_name is not None:
                            return preference_name, object_types

        elif isinstance(scoring_expression, tuple) or isinstance(scoring_expression, list):
            for item in scoring_expression:
                preference_name, object_types = self._extract_name_and_types(item)

                if preference_name is not None:
                    return preference_name, object_types

        else:
            return None, None

        raise ValueError(f"No name found in scoring expression!")

    def _describe_setup(self, setup_ast: tatsu.ast.AST, condition_type: typing.Optional[str] = None):
        '''
        Describe the setup of the game, including conditions that need to be satisfied once (game-optional)
        and conditions that must be met continually (game-conserved)
        '''

        rule = setup_ast["parseinfo"].rule  # type: ignore

        if rule == "setup":
            return self._describe_setup(typing.cast(tatsu.ast.AST, setup_ast["setup"]), condition_type)

        elif rule == "setup_statement":
            return self._describe_setup(typing.cast(tatsu.ast.AST, setup_ast["statement"]), condition_type)

        elif rule == "super_predicate":
            return self._describe_predicate(typing.cast(tatsu.ast.AST, setup_ast)), condition_type

        elif rule == "setup_not":
            text, condition_type = self._describe_setup(setup_ast["not_args"], condition_type) # type: ignore
            return f"it's not the case that {text}", condition_type # type: ignore

        elif rule == "setup_and":

            description = ""
            conditions_and_types = [self._describe_setup(sub, condition_type) for sub in setup_ast["and_args"]] # type: ignore

            optional_conditions = [condition for condition, condition_type in conditions_and_types if condition_type == "optional"] # type: ignore
            if len(optional_conditions) > 0:
                description += "the following must all be true for at least one time step:"
                description += "\n- " + "\n- ".join(optional_conditions)

            # We'll default to calling conditions with ambiguous types "conserved"
            conserved_conditions = [condition for condition, condition_type in conditions_and_types if condition_type == "conserved" or condition_type is None] # type: ignore
            if len(conserved_conditions) > 0:
                if len(optional_conditions) > 0:
                    description += "\n\nand in addition, "

                description += "the following must all be true for every time step:"
                description += "\n- " + "\n- ".join(conserved_conditions)

            return description, None

        elif rule == "setup_or":

            description = ""
            conditions_and_types = [self._describe_setup(sub, condition_type) for sub in setup_ast["or_args"]] # type: ignore

            optional_conditions = [condition for condition, condition_type in conditions_and_types if condition_type == "optional"] # type: ignore
            if len(optional_conditions) > 0:
                description += "at least one of the following must be true for at least one time step:"
                description += "\n- " + "\n- ".join(optional_conditions)

            # We'll default to calling conditions with ambiguous types "conserved"
            conserved_conditions = [condition for condition, condition_type in conditions_and_types if condition_type == "conserved" or condition_type is None] # type: ignore
            if len(conserved_conditions) > 0:
                if len(optional_conditions) > 0:
                    description += "\n\nand in addition, "

                description += "at least one of the following must be true for every time step:"
                description += "\n- " + "\n- ".join(conserved_conditions)

            return description, None

        elif rule == "setup_exists":
            variable_type_mapping = extract_variable_type_mapping(setup_ast["exists_vars"]["variables"]) # type: ignore

            def group_func(key):
                return ";".join(variable_type_mapping[key])

            new_variables = []
            for key, group in groupby(sorted(variable_type_mapping.keys(), key=group_func), key=group_func):
                group = list(group)
                if len(group) == 1:
                    new_variables.append(f"an object {group[0]} of type {self.engine.join(key.split(';'), conj='or')}")
                else:
                    new_variables.append(f"objects {self.engine.join(group)} of type {self.engine.join(key.split(';'), conj='or')}")

            text, condition_type = self._describe_setup(setup_ast["exists_args"], condition_type) # type: ignore

            return f"there exists {self.engine.join(new_variables)}, such that {text}", condition_type # type: ignore

        elif rule == "setup_forall":
            variable_type_mapping = extract_variable_type_mapping(setup_ast["forall_vars"]["variables"]) # type: ignore

            new_variables = []
            for var, types in variable_type_mapping.items():
                new_variables.append(f"object {var} of type {self.engine.join(types, conj='or')}")

            text, condition_type = self._describe_setup(setup_ast["forall_args"], condition_type) # type: ignore

            return f"for any {self.engine.join(new_variables)}, {text}", condition_type # type: ignore

        elif rule == "setup_game_optional":
            text, _ = self._describe_setup(setup_ast["optional_pred"], condition_type) # type: ignore
            return text, "optional"

        elif rule == "setup_game_conserved":
            text, _ = self._describe_setup(setup_ast["conserved_pred"], condition_type) # type: ignore
            return text, "conserved"

        else:
            raise ValueError(f"Unknown setup expression rule: {rule}")

    def _describe_preference(self, preference_ast: tatsu.ast.AST):
        '''
        Describe a particular preference of game, calling out whether it uses an external forall
        '''

        description = ""

        pref_def = typing.cast(tatsu.ast.AST, preference_ast["definition"])
        rule = pref_def["parseinfo"].rule   # type: ignore

        if rule == "preference":
            name = typing.cast(str, pref_def["pref_name"])
            description += f"-----Preference {self.preference_index + 1}-----"
            self.preference_name_to_index[name] = self.preference_index + 1

            body = pref_def["pref_body"]["body"] # type: ignore

            description += self._indent(self._describe_preference_body(body))
            self.preference_index += 1

        # This case handles externall forall preferences
        elif rule == "pref_forall":

            forall_vars = pref_def["forall_vars"]
            forall_pref = pref_def["forall_pref"]

            variable_type_mapping = extract_variable_type_mapping(forall_vars["variables"])  # type: ignore

            sub_preferences = forall_pref["preferences"] # type: ignore
            if isinstance(sub_preferences, tatsu.ast.AST):
                sub_preferences = [sub_preferences]

            for sub_idx, sub_preference in enumerate(sub_preferences):
                name = typing.cast(str, sub_preference["pref_name"])

                self.external_forall_preference_mappings[name] = variable_type_mapping # type: ignore

                newline = '\n' if sub_idx > 0 else ''
                description += f"{newline}-----Preference {self.preference_index + 1}-----"
                self.preference_name_to_index[name] = self.preference_index + 1

                body = sub_preference["pref_body"]["body"] # type: ignore

                description += self._indent(self._describe_preference_body(body, variable_type_mapping))
                self.preference_index += 1

        return description

    def _describe_preference_body(self, body_ast: tatsu.ast.AST, additional_variable_mapping: typing.Dict[str, typing.List[str]] = {}):
        '''
        Describe the main body of a preference (i.e. the part after any external-foralls / names). Optionally, additional variable
        mappings from an external forall can be passed in to be used in the description
        '''

        description = ""

        if body_ast.parseinfo.rule == "pref_body_exists":

            variable_type_mapping = extract_variable_type_mapping(body_ast["exists_vars"]["variables"])
            description += "\nThe variables required by this preference are:"

            def group_func_external(key):
                return ";".join(additional_variable_mapping[key])

            def group_func(key):
                return ";".join(variable_type_mapping[key])

            for key, group in groupby(sorted(additional_variable_mapping.keys(), key=group_func_external), key=group_func_external):
                group = list(group)
                description += f"\n-{self.engine.join(group)} of type {self.engine.join(key.split(';'), conj='or')}"

            for key, group in groupby(sorted(variable_type_mapping.keys(), key=group_func), key=group_func):
                group = list(group)
                description += f"\n-{self.engine.join(group)} of type {self.engine.join(key.split(';'), conj='or')}"

            temporal_predicate_ast = body_ast["exists_args"]

        # These cases handle preferences that don't have any variables quantified with an exists (e.g. they're all from an external forall)
        elif body_ast.parseinfo.rule == "then":
            description += "\nThe variables required by this preference are:"

            def group_func_external(key):
                return ";".join(additional_variable_mapping[key])

            for key, group in groupby(sorted(additional_variable_mapping.keys(), key=group_func_external), key=group_func_external):
                group = list(group)
                description += f"\n-{self.engine.join(group)} of type {self.engine.join(key.split(';'), conj='or')}"

            temporal_predicate_ast = body_ast

        elif body_ast.parseinfo.rule == "at_end":
            description += "\nThe variables required by this preference are:"

            def group_func_external(key):
                return ";".join(additional_variable_mapping[key])

            for key, group in groupby(sorted(additional_variable_mapping.keys(), key=group_func_external), key=group_func_external):
                group = list(group)
                description += f"\n-{self.engine.join(group)} of type {self.engine.join(key.split(';'), conj='or')}"

            temporal_predicate_ast = body_ast

        else:
            raise NotImplementedError(f"Unknown preference body rule: {body_ast.parseinfo.rule}")

        description += "\n\nThis preference is satisfied when:"

        if temporal_predicate_ast.parseinfo.rule == "at_end":
            description += f"\n- in the final game state, {self._describe_predicate(temporal_predicate_ast['at_end_pred'])}" # type: ignore

        elif temporal_predicate_ast.parseinfo.rule == "then":

            temporal_predicates = [func['seq_func'] for func in temporal_predicate_ast["then_funcs"]]
            for idx, temporal_predicate in enumerate(temporal_predicates):
                if len(temporal_predicates) == 1:
                    description += "\n- "
                elif idx == 0:
                    description += f"\n- first, "
                elif idx == len(temporal_predicates) - 1:
                    description += f"\n- finally, "
                else:
                    description += f"\n- next, "

                temporal_type = self._predicate_type(temporal_predicate)
                if temporal_type == PredicateType.ONCE:
                    description += f"there is a state where {self._describe_predicate(temporal_predicate['once_pred'])}"

                elif temporal_type == PredicateType.ONCE_MEASURE:
                    description += f"there is a state where {self._describe_predicate(temporal_predicate['once_measure_pred'])}."
                    description += f" In addition, measure and record {self._describe_predicate(temporal_predicate['measurement'])}"

                elif temporal_type == PredicateType.HOLD:
                    description += f"there is a sequence of one or more states where {self._describe_predicate(temporal_predicate['hold_pred'])}"

                elif temporal_type == PredicateType.HOLD_WHILE:
                    description += f"there is a sequence of one or more states where {self._describe_predicate(temporal_predicate['hold_pred'])}"

                    if isinstance(temporal_predicate["while_preds"], list):
                        while_desc = self.engine.join(['a state where (' + self._describe_predicate(pred) + ')' for pred in temporal_predicate['while_preds']])
                        description += f" Additionally, during this sequence there is {while_desc} (in that order)."
                    else:
                        description += f" Additionally, during this sequence there is  a state where ({self._describe_predicate(temporal_predicate['while_preds'])})."

        else:
            raise ValueError(f"Unknown body exist-args rule: {temporal_predicate_ast.parseinfo.rule}")

        return description


    def _describe_predicate(self, predicate: tatsu.ast.AST):

        rule = predicate["parseinfo"].rule # type: ignore

        if rule == "predicate":

            name = extract_predicate_function_name(predicate)
            variables = extract_variables(predicate)

            # Special case for predicates that can have a variable number of arguments
            if name == "adjacent_side":
                name += f"_{len(variables)}_args"

            # Issue: if a predicate uses the same variable in both arguments, extract_variables will only return it once
            # We can sidestep this by copying out variables to the right length
            if len(variables) < PREDICATE_DESCRIPTIONS[name].count('{'): # type: ignore
                diff = PREDICATE_DESCRIPTIONS[name].count('{') - len(variables) # type: ignore
                variables += [variables[-1]] * diff

            return PREDICATE_DESCRIPTIONS[name].format(*variables)

        elif rule == "super_predicate":
            return self._describe_predicate(predicate["pred"]) # type: ignore

        elif rule == "super_predicate_not":
            return f"it's not the case that {self._describe_predicate(predicate['not_args'])}" # type: ignore

        elif rule == "super_predicate_and":
            return self.engine.join(["(" + self._describe_predicate(sub) + ")" for sub in predicate["and_args"]]) # type: ignore

        elif rule == "super_predicate_or":
            return self.engine.join(["(" + self._describe_predicate(sub) + ")" for sub in predicate["or_args"]], conj="or") # type: ignore

        elif rule == "super_predicate_exists":
            variable_type_mapping = extract_variable_type_mapping(predicate["exists_vars"]["variables"]) # type: ignore

            def group_func(key):
                return ";".join(variable_type_mapping[key])

            new_variables = []
            for key, group in groupby(sorted(variable_type_mapping.keys(), key=group_func), key=group_func):
                group = list(group)
                if len(group) == 1:
                    new_variables.append(f"an object {group[0]} of type {self.engine.join(key.split(';'), conj='or')}")
                else:
                    new_variables.append(f"objects {self.engine.join(group)} of type {self.engine.join(key.split(';'), conj='or')}")

            return f"there exists {self.engine.join(new_variables)}, such that {self._describe_predicate(predicate['exists_args'])}" # type: ignore

        elif rule == "super_predicate_forall":
            variable_type_mapping = extract_variable_type_mapping(predicate["forall_vars"]["variables"]) # type: ignore

            new_variables = []
            for var, types in variable_type_mapping.items():
                new_variables.append(f"object {var} of type {self.engine.join(types, conj='or')}")

            return f"for any {self.engine.join(new_variables)}, {self._describe_predicate(predicate['forall_args'])}" # type: ignore

        elif rule == "function_comparison":

            # Special case for multi-arg equality
            if predicate["comp"].parseinfo.rule == "multiple_args_equal_comparison":
                comp_args = [arg["arg"] for arg in predicate["comp"]["equal_comp_args"]] # type: ignore
                comp_descriptions = [self._describe_predicate(arg) if isinstance(arg, tatsu.ast.AST) else arg for arg in comp_args]
                description = f"{self.engine.join(comp_descriptions)} are all equal"

                return description

            comparison_operator = predicate["comp"]["comp_op"] # type: ignore

            comp_arg_1 = predicate["comp"]["arg_1"]["arg"] # type: ignore
            comp_arg_2 = predicate["comp"]["arg_2"]["arg"] # type: ignore

            if isinstance(comp_arg_1, tatsu.ast.AST):
                comp_arg_1 = self._describe_predicate(comp_arg_1)

            if isinstance(comp_arg_2, tatsu.ast.AST):
                comp_arg_2 = self._describe_predicate(comp_arg_2)

            if comparison_operator == "=":
                return f"{comp_arg_1} is equal to {comp_arg_2}"
            elif comparison_operator == "<":
                return f"{comp_arg_1} is less than {comp_arg_2}"
            elif comparison_operator == "<=":
                return f"{comp_arg_1} is less than or equal to {comp_arg_2}"
            elif comparison_operator == ">":
                return f"{comp_arg_1} is greater than {comp_arg_2}"
            elif comparison_operator == ">=":
                return f"{comp_arg_1} is greater than or equal to {comp_arg_2}"

        elif rule == "function_eval":
            name = extract_predicate_function_name(predicate)
            variables = extract_variables(predicate)

            return FUNCTION_DESCRIPTIONS[name].format(*variables)

        elif rule in ("comparison_arg_number_value", "time_number_value", "score_number_value", "pref_count_number_value", "scoring_number_value"):
            return predicate["terminal"] # type: ignore

        else:
            raise ValueError(f"Error: Unknown rule '{rule}'")

        return ''

    def _describe_terminal(self, terminal_ast: typing.Optional[tatsu.ast.AST]):
        '''
        Determine whether the terminal conditions of the game have been met
        '''
        if terminal_ast is None:
            return False

        rule = terminal_ast["parseinfo"].rule  # type: ignore

        if rule == "terminal":
            return self._describe_terminal(terminal_ast["terminal"])

        elif rule == "terminal_not":
            return f"it's not the case that {self._describe_terminal(terminal_ast['not_args'])}" # type: ignore

        elif rule == "terminal_and":
            return self.engine.join(["(" + self._describe_terminal(sub) + ")" for sub in terminal_ast["and_args"]]) # type: ignore

        elif rule == "terminal_or":
            return self.engine.join(["(" + self._describe_terminal(sub) + ")" for sub in terminal_ast["or_args"]], conj="or") # type: ignore

        elif rule == "terminal_comp":
            terminal_ast = typing.cast(tatsu.ast.AST, terminal_ast["comp"])
            comparison_operator = terminal_ast["op"]

            expr_1 = self._describe_scoring(terminal_ast["expr_1"]) # type: ignore
            expr_2 = self._describe_scoring(terminal_ast["expr_2"]) # type: ignore

            if comparison_operator == "=":
                return f"{expr_1} is equal to {expr_2}" # type: ignore
            elif comparison_operator == "<":
                return f"{expr_1} is less than {expr_2}" # type: ignore
            elif comparison_operator == "<=":
                return f"{expr_1} is less than or equal to {expr_2}" # type: ignore
            elif comparison_operator == ">":
                return f"{expr_1} is greater than {expr_2}" # type: ignore
            elif comparison_operator == ">=":
                return f"{expr_1} is greater than or equal to {expr_2}" # type: ignore

        elif rule in ("comparison_arg_number_value", "time_number_value", "score_number_value", "pref_count_number_value", "scoring_number_value"):
            return terminal_ast["terminal"] # type: ignore

        else:
            raise ValueError(f"Error: Unknown terminal rule '{rule}'")

    def _external_scoring_description(self, preference_name, external_object_types):
        '''
        A helper function for describing the special scoring syntax in which variable
        types are passed with colons after the preference name
        '''
        if external_object_types is None:
            return ""

        # Issue: it's possible for regrown games to apply the external scoring syntax to games without and
        # external forall. In this case, I suppose we'll just ignore it? (Other options are adding an error message)
        if preference_name not in self.external_forall_preference_mappings:
            return ""


        specified_variables = list(self.external_forall_preference_mappings[preference_name].keys())[:len(external_object_types)] # type: ignore
        mapping_description = self.engine.join([f"{var} is bound to an object of type {var_type}" for var, var_type in zip(specified_variables, external_object_types)])

        return f", where {mapping_description}"

    def _describe_scoring(self, scoring_ast: typing.Optional[tatsu.ast.AST]) -> str:

        if isinstance(scoring_ast, str):
            return scoring_ast

        scoring_ast = typing.cast(tatsu.ast.AST, scoring_ast)

        rule = scoring_ast["parseinfo"].rule  # type: ignore

        if rule in ("scoring_expr", "scoring_expr_or_number"):
            return self._describe_scoring(scoring_ast["expr"]) # type: ignore

        elif rule == "scoring_multi_expr":
            operator = scoring_ast["op"] # type: ignore
            expressions = scoring_ast["expr"] # type: ignore

            if isinstance(expressions, tatsu.ast.AST):
                return self._describe_scoring(expressions)

            elif isinstance(expressions, list):
                if operator == "+":
                    return f"the sum of {self.engine.join([f'({self._describe_scoring(expression)})' for expression in expressions])}"

                elif operator == "*":
                    return f"the product of {self.engine.join([f'({self._describe_scoring(expression)})' for expression in expressions])}"

        elif rule == "scoring_binary_expr":
            operator = scoring_ast["op"] # type: ignore

            expr_1 = self._describe_scoring(scoring_ast["expr_1"]) # type: ignore
            expr_2 = self._describe_scoring(scoring_ast["expr_2"]) # type: ignore

            if operator == "-":
                return f"{expr_1} minus {expr_2}"
            elif operator == "/":
                return f"{expr_1} divided by {expr_2}"

        elif rule == "scoring_neg_expr":
            return f"negative {self._describe_scoring(scoring_ast['expr'])}" # type: ignore

        elif rule == "scoring_comparison":
            comparison_operator = scoring_ast["comp"]["op"] # type: ignore

            # Issue: occasional syntax errors can cause the operator to be invalid / not appear
            if comparison_operator is None:
                return "[SYNTAX ERROR IN SCORING]"

            expr_1 = self._describe_scoring(scoring_ast["comp"]["expr_1"]) # type: ignore
            expr_2 = self._describe_scoring(scoring_ast["comp"]["expr_2"]) # type: ignore

            if comparison_operator == "=":
                return f"{expr_1} is equal to {expr_2}" # type: ignore
            elif comparison_operator == "<":
                return f"{expr_1} is less than {expr_2}" # type: ignore
            elif comparison_operator == "<=":
                return f"{expr_1} is less than or equal to {expr_2}" # type: ignore
            elif comparison_operator == ">":
                return f"{expr_1} is greater than {expr_2}" # type: ignore
            elif comparison_operator == ">=":
                return f"{expr_1} is greater than or equal to {expr_2}" # type: ignore

        elif rule == "preference_eval":
            return self._describe_scoring(scoring_ast["count_method"]) # type: ignore

        elif rule == "scoring_external_maximize":
            # Extracts the name of the first predicate encountered -- fine since they all share an external forall
            preference_name, _ = self._extract_name_and_types(scoring_ast) # type: ignore
            external_variable_mapping = self.external_forall_preference_mappings[preference_name] # type: ignore
            quanitification_string = self.engine.join([f"{var} (of type {self.engine.join(types, conj='or')})" for var, types in external_variable_mapping.items()]) # type: ignore

            internal_description = self._describe_scoring(scoring_ast["scoring_expr"]) # type: ignore
            return f"the maximum value of ({internal_description}) over all quantifications of {quanitification_string}"

        elif rule == "scoring_external_minimize":
            preference_name, _ = self._extract_name_and_types(scoring_ast) # type: ignore
            external_variables = self.external_forall_preference_mappings[preference_name] # type: ignore
            quanitification_string = self.engine.join([f"{var} (of type {self.engine.join(types, conj='or')})" for var, types in external_variable_mapping.items()]) # type: ignore

            internal_description = self._describe_scoring(scoring_ast["scoring_expr"]) # type: ignore
            return f"the minimum value of ({internal_description}) over all quantifications of {quanitification_string}"

        elif rule == "count":
            preference_name, object_types = self._extract_name_and_types(scoring_ast) # type: ignore
            external_scoring_desc = self._external_scoring_description(preference_name, object_types)

            return f"the number of times '{preference_name}' has been satisfied" + external_scoring_desc

        elif rule == "count_overlapping":
            preference_name, object_types = self._extract_name_and_types(scoring_ast) # type: ignore
            external_scoring_desc = self._external_scoring_description(preference_name, object_types)

            return f"the number of times '{preference_name}' has been satisfied in overlapping intervals" + external_scoring_desc

        elif rule == "count_once":
            preference_name, object_types = self._extract_name_and_types(scoring_ast) # type: ignore
            external_scoring_desc = self._external_scoring_description(preference_name, object_types)
            # return f"whether '{preference_name}' has been satisfied at least once" + external_scoring_desc
            return f"min(1, the number of times '{preference_name}' has been satisfied{external_scoring_desc})"

        elif rule == "count_once_per_objects":
            preference_name, object_types = self._extract_name_and_types(scoring_ast) # type: ignore
            external_scoring_desc = self._external_scoring_description(preference_name, object_types)

            return f"the number of times '{preference_name}' has been satisfied with different objects" + external_scoring_desc

        elif rule == "count_measure":
            preference_name, object_types = self._extract_name_and_types(scoring_ast) # type: ignore

            if object_types is None:
                return f"the sum of all values measured during satisfactions of '{preference_name}'"
            else:
                raise ValueError("Error: count_measure does not support specific object types (I think?)")

        elif rule == "count_unique_positions":
            preference_name, object_types = self._extract_name_and_types(scoring_ast) # type: ignore
            external_scoring_desc = self._external_scoring_description(preference_name, object_types)

            return f"the number of times '{preference_name}' has been satisfied with stationary objects in different positions" + external_scoring_desc

        elif rule == "count_same_positions":
            preference_name, object_types = self._extract_name_and_types(scoring_ast) # type: ignore
            external_scoring_desc = self._external_scoring_description(preference_name, object_types)

            return f"the maximum number of satisfactions of '{preference_name}' where stationary objects remain in the same place between satisfactions" + external_scoring_desc

        elif rule == "count_once_per_external_objects":
            preference_name, object_types = self._extract_name_and_types(scoring_ast) # type: ignore

            external_variables = list(self.external_forall_preference_mappings[preference_name].keys()) # type: ignore
            external_scoring_desc = self._external_scoring_description(preference_name, object_types)

            return f"the number of times '{preference_name}' has been satisfied with different quantifications of {self.engine.join(external_variables, conj='and')}" + external_scoring_desc

        # TODO: is this correct?
        elif rule == "count_shortest":
            preference_name, object_types = self._extract_name_and_types(scoring_ast) # type: ignore
            external_scoring_desc = self._external_scoring_description(preference_name, object_types)

            return f"the number of times '{preference_name}' has been satisfied in the shortest possible sequence of states" + external_scoring_desc

        elif rule in ("comparison_arg_number_value", "time_number_value", "score_number_value", "pref_count_number_value", "scoring_number_value"):
            return scoring_ast["terminal"]  # type: ignore

        else:
            raise ValueError(f"Error: Unknown rule '{rule}' in scoring expression")

    def describe_stage_0(self, game_text_or_ast: typing.Union[str, tatsu.ast.AST], format_for_html: bool = False):
        '''
        Returns the raw code of each section of the game text or AST. Optionally format the
        descriptions for display in an HTML page
        '''
        if isinstance(game_text_or_ast, str):
            game_ast = typing.cast(tatsu.ast.AST, self.grammar_parser.parse(game_text_or_ast))
        else:
            game_ast = game_text_or_ast

        if format_for_html:
            delimiter = '<br>'
            formatting = "<pre><code>{0}</code></pre>"
        else:
            delimiter = '\n'
            formatting = "{0}"

        setup_description = ""
        preferences_description = ""
        terminal_description = ""
        scoring_description = ""

        for section in game_ast:
            if not isinstance(section, tuple):
                continue

            if section[0] == SETUP:
                setup_description = formatting.format(ast_section_to_string(section, SETUP, delimiter))
            elif section[0] == PREFERENCES:
                preferences_description = formatting.format(ast_section_to_string(section, PREFERENCES, delimiter))
            elif section[0] == TERMINAL:
                terminal_description = formatting.format(ast_section_to_string(section, TERMINAL, delimiter))
            elif section[0] == SCORING:
                scoring_description = formatting.format(ast_section_to_string(section, SCORING, delimiter))

        return setup_description, preferences_description, terminal_description, scoring_description

    def describe_stage_1(self, game_text_or_ast: typing.Union[str, tatsu.ast.AST]):
        '''
        Generate a "stage 1" description of the provided game text or AST. A stage 1 description is
        templated based on a series of hard-coded rules. Description will be split by game section
        (setup, preferences, terminal, and scoring)
        '''

        if isinstance(game_text_or_ast, str):
            game_ast = typing.cast(tatsu.ast.AST, self.grammar_parser.parse(game_text_or_ast))
        else:
            game_ast = game_text_or_ast

        game_info = {}
        self._extract_game_info(game_ast, game_info)

        self.preference_index = 0
        self.preference_name_to_index = {}
        self.external_forall_preference_mappings = {}

        if game_info.get("setup") is not None:
            setup_description, _ = self._describe_setup(game_info["setup"])
            setup_description = SETUP_HEADER + setup_description
        else:
            setup_description = ""

        if game_info.get("preferences") is not None:
            preferences_description = PREFERENCES_HEADER
            for idx, preference in enumerate(game_info["preferences"][0]):
                preferences_description +=  f"\n\n{self._describe_preference(preference)}"
        else:
            preferences_description = ""

        if game_info.get("terminal") is not None:
            terminal_description = TERMINAL_HEADER + self._describe_terminal(game_info["terminal"])

            # Replace specific preference names with their index
            for name, index in self.preference_name_to_index.items():
                terminal_description = terminal_description.replace(name, f"Preference {index}")

        else:
            terminal_description = ""

        if game_info.get("scoring") is not None:
            scoring_description = SCORING_HEADER + self._describe_scoring(game_info["scoring"])

            # Replace specific preference names with their index
            for name, index in self.preference_name_to_index.items():
                scoring_description = scoring_description.replace(name, f"Preference {index}")

        else:
            scoring_description = ""

        return setup_description, preferences_description, terminal_description, scoring_description

    def describe_stage_2(self, game_text_or_ast: typing.Union[str, tatsu.ast.AST], stage_1_descriptions: typing.Optional[tuple] = None):
        '''
        Generate a "stage 2" description of the provided game text or AST. A stage 2 description uses
        an LLM to convert a stage 1 description into more naturalistic language
        '''

        assert self.openai_model_str is not None, "Error: No OpenAI model specified provided for stage 2 description"

        if stage_1_descriptions is None:
            setup_stage_1, preferences_stage_1, terminal_stage_1, scoring_stage_1 = self.describe_stage_1(game_text_or_ast)
        else:
            setup_stage_1, preferences_stage_1, terminal_stage_1, scoring_stage_1 = stage_1_descriptions


        # Generate prompts
        all_prompts = compile_prompts_from_data(initial_stage=1, final_stage=2,
                                                translations_path="./selected_human_and_map_elites_translations.csv")
        setup_prompt, preferences_prompt, terminal_prompt, scoring_prompt, _ = all_prompts

        if setup_stage_1 != "":
            setup_description = self._query_openai(setup_prompt.format(setup_stage_1))
        else:
            setup_description = ""

        if preferences_stage_1 != "":
            preferences_description = self._query_openai(preferences_prompt.format(preferences_stage_1))
        else:
            preferences_description = ""

        if terminal_stage_1 != "":
            terminal_description = self._query_openai(terminal_prompt.format(terminal_stage_1))
        else:
            terminal_description = ""

        if scoring_stage_1 != "":
            scoring_description = self._query_openai(scoring_prompt.format(scoring_stage_1))
        else:
            scoring_description = ""

        return setup_description, preferences_description, terminal_description, scoring_description

    def describe_stage_3(self, game_text_or_ast: typing.Union[str, tatsu.ast.AST], stage_2_descriptions: typing.Optional[tuple] = None):
        '''
        Generate a "stage 3" description of the provided game text or AST. A stage 3 description uses an LLM
        to convert a stage 2 description (consisting of independent section descriptions) into a single, short
        natural language description of the game
        '''

        assert self.openai_model_str is not None, "Error: No OpenAI model specified provided for stage 2 description"

        if stage_2_descriptions is None:
            setup_stage_2, preferences_stage_2, terminal_stage_2, scoring_stage_2 = self.describe_stage_1(game_text_or_ast)
        else:
            setup_stage_2, preferences_stage_2, terminal_stage_2, scoring_stage_2 = stage_2_descriptions


        _, _, _, _, overall_prompt = compile_prompts_from_data(initial_stage=2, final_stage=3,
                                                               translations_path="./selected_human_and_map_elites_translations.csv")

        setup_prefix = "" if setup_stage_2 == "" else "\n\n"
        terminal_prefix = "" if terminal_stage_2 == "" else "\n\n"

        game_description = f"{setup_prefix}{setup_stage_2}\n\n{preferences_stage_2}{terminal_prefix}{terminal_stage_2}\n\n{scoring_stage_2}"

        stage_3_description = self._query_openai(overall_prompt.format(game_description))

        return stage_3_description

    def _prepare_data_for_html_display(self, descriptions_by_stage: typing.List[typing.Tuple[str, str, str, str]]):
        '''
        Helper function for formatting the descriptions for display in an HTML page
        '''
        grouped_descriptions = list(zip(*descriptions_by_stage))

        columns = [f"Stage {i}" for i in range(len(descriptions_by_stage))]

        # The content of the table in HTML (for just the current game)
        table_html = tabulate.tabulate(grouped_descriptions, headers=columns, tablefmt="unsafehtml")

        # Replace all newlines inside the <tbody> tags with <br>
        tbody_start = table_html.find("<tbody>")
        tbody_end = table_html.find("</tbody>")
        table_html = table_html[:tbody_start] + table_html[tbody_start:tbody_end].replace("\n", "<br>") + table_html[tbody_end:]

        table_html = table_html.replace('<table>', '<table class="table table-striped table-bordered">')
        table_html = table_html.replace('<thead>', '<thead class="thead-dark">')
        table_html = table_html.replace('<th>', '<th scope="col">')

        return table_html

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--description_stage", type=int, default=1, help="The maximum 'stage' of description to generate for each game")
    parser.add_argument("--gpt_model", type=str, default="gpt-3.5-turbo", choices=["gpt-3.5-turbo", "gpt-4"])
    parser.add_argument("--games_path", type=str, default=DEFAULT_GAMES_PATH)
    parser.add_argument("--output_path", type=str, default=".")

    args = parser.parse_args()

    grammar = open(DEFAULT_GRAMMAR_PATH).read()
    grammar_parser = tatsu.compile(grammar)
    game_asts = list(cached_load_and_parse_games_from_file(args.games_path, grammar_parser, False, relative_path='.'))
    game_describer = GameDescriber(openai_model_str=args.gpt_model)

    input_file_name = os.path.splitext(os.path.basename(args.games_path))[0]

    game_table_htmls = []
    for game in tqdm(game_asts, desc="Generating game descriptions"):
        descriptions_by_stage = []

        stage_0_descriptions = game_describer.describe_stage_0(game, format_for_html=True)
        descriptions_by_stage.append(stage_0_descriptions)

        if args.description_stage >= 1:
            stage_1_descriptions = game_describer.describe_stage_1(game)
            descriptions_by_stage.append(stage_1_descriptions)

        if args.description_stage >= 2:
            stage_2_descriptions = game_describer.describe_stage_2(game, stage_1_descriptions)
            descriptions_by_stage.append(stage_2_descriptions)

        if args.description_stage >= 3:
            stage_3_description = game_describer.describe_stage_3(game, stage_2_descriptions)
            descriptions_by_stage.append((stage_3_description, "", "", ""))

        # The content of the table in HTML (for just the current game)
        table_html = game_describer._prepare_data_for_html_display(descriptions_by_stage)

        game_table_htmls.append(table_html)
        joined_html = "\n".join(game_table_htmls)

        # The ultimate html to be saved / displayed
        full_html = TABLE_HTML_TEMPLATE.format(STYLE_HTML, joined_html)

        output_filename = os.path.join(args.output_path, f"{input_file_name}_game_descriptions_stage_{args.description_stage}_model_{args.gpt_model}.html")
        with open(output_filename, "w") as file:
            file.write(full_html)

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

import flask
import libs.utils as utils
import openai
import requests
from jinja2 import Template
from libs.base import Base
from libs.recursive_search import search_dicts
from models.gpt_models import (
    BaseRequestBody,
    Function,
    FunctionCall,
    FunctionRequestBody,
    Message,
)
from models.service import Domain
from models.state import State

from gpt_assistant.libs.base import Base
from gpt_assistant.libs.utils import dictify_list_of_base_models

logger = utils.get_logger(__name__, level="DEBUG")


class GPTInterface(Base):
    def __init__(self):
        self.system_prompt = """
        You are an AI assistant with 
        control over my Home Assistant system.
        You collaborate with other AI helpers.
        Use your function call ability to 
        accomplish the user's request. 
        Only use the functions and data that are available
        to you. You can use the functions to change states
        or get data from the system. When a request is complete,
        start message with "Request complete: "

        First, come up with a plan, then execute it in the
        following order (each step will be a separate message):
        1. Choose a domain of service functions to use
        2. Make a function call
        3. Evaluate the result
        4. Repeat steps 1-3 until the user's request is complete.
        5. Tell the user the result of the request.

        Here's the request:\n
        """

    def execute_request(self, request: str) -> str:
        """Execute a user's request"""

        success = False
        retries = 0
        max_retries = 3
        request_body = None
        while not success and retries < max_retries:
            if request_body is None:
                evaluation, req_body = self.execute_first_three_steps(request)
            else:
                evaluation, req_body = self.execute_first_three_steps(
                    request, request_body
                )
            if evaluation.startswith("Request complete: "):
                success = True
                return evaluation
            else:
                request_body = req_body

    def execute_first_three_steps(
        self, request: str, request_body: Union[BaseRequestBody, None] = None
    ) -> Tuple[List[State], BaseRequestBody]:
        # stage 1: choose domain
        domain, request_body = self._choose_domain(request)

        # stage 2: make function call
        state_changes, request_body = self._make_function_call(request_body)

        # stage 3: evaluate result
        evaluation = self._evaluate_result(state_changes, request_body)
        return evaluation, request_body

    def _evaluate_result(
        self, state_changes: List[State], request_body: BaseRequestBody
    ) -> Tuple[List[State], BaseRequestBody]:
        """Evaluate the result of the function call"""
        logger.debug("In _evaluate_result")

        # get state changes
        state_changes = self._get_state_changes(state_changes)

        # record function call message
        request_body.messages.append(
            Message(
                role="home_assistant",
                content=f"function response (state changes): {dictify_list_of_base_models(state_changes)}",
            ),
        )

        response = openai.Completion.create(
            **request_body.dict(),
        )

        return response["choices"][0]["message"]["content"]

    def _make_function_call(
        self, request_body: BaseRequestBody
    ) -> Tuple[List[State], FunctionRequestBody]:
        """Request a function call from AI assistant"""
        logger.debug("In _make_function_call")

        domain = search_dicts(
            dictify_list_of_base_models(request_body.messages),
            "assistant",
        )[-1]["content"]
        logger.debug(f"domain: {domain}")

        # get functions
        functions = self._get_functions(domain)

        request_body_dict = request_body.dict()
        request_body_dict["functions"] = functions
        request_body = FunctionRequestBody(**request_body_dict)

        # get function call from openai
        response = openai.Completion.create(
            **request_body.dict(),
        )
        logger.debug(f"response: {response}")

        if response.get("function_call"):
            fn_call = FunctionCall(**response["choices"][0]["message"]["function_call"])
            logger.debug(f"fn_call: {fn_call}")

            state_changes = self._call_function(fn_call)

            # record function call message
            request_body.messages.append(
                Message(
                    role="assistant",
                    content=fn_call.dict(),
                ),
                Message(
                    role="home_assistant",
                    content=f"state changes: {dictify_list_of_base_models(state_changes)}",
                ),
            )

            return state_changes, request_body
        else:
            raise Exception("No function call returned from OpenAI")

    def _call_function(self, function_call: FunctionCall) -> List[State]:
        """
        Call a function and return the state
        that results from the function call
        """
        logger.debug("In _call_function")

        domain_name = function_call.name.split("__")[0]
        service_name = function_call.name.split("__")[1]
        args = function_call.arguments_dict
        fn = self._get_function_from_name(function_call.name)
        domain_ = self._get_domain_from_name(domain_name)

        url = f"{self.home_assistant_url}/api/services/{domain_name}/{service_name}"
        headers = self.home_assistant_headers

        # get data from home assistant
        response = requests.post(
            url,
            headers=headers,
            json=json.loads(function_call.arguments),
        )
        logger.debug(f"response: {response.json()}")

        states: List[State] = []
        for state_dict in response.json():
            state = State(**state_dict)
            states.append(state)
        logger.debug(f"states: {states}")

        return states

    def get_domain_from_name(self, domain_name: str) -> Domain:
        """Get domain from name"""
        logger.debug("In _get_domain_from_name")

        # get domains
        domains = self._get_domains()

        for domain in domains:
            if domain.name == domain_name:
                return domain

        raise Exception(f"Domain {domain_name} not found")

    def _get_domains_from_functions(self, functions: List[Function]) -> List[Domain]:
        """Get domains from functions using jinja2 template"""

        grouped = defaultdict(list)
        for function in functions:
            domain = function.name.split("__")[0]
            grouped[domain].append(function)

        with open("functions_to_domains.jinja2") as f:
            template = Template(f.read())

        domains = template.render(grouped=grouped)

        return domains

    def _choose_domain(
        self, request: str, request_body: Union[None, BaseRequestBody] = None
    ) -> Tuple[str, BaseRequestBody]:
        """Choose a domain for the request"""

        # get domains
        domains = self._get_domains()

        # domain content
        domain_content = """
        Here's a list of domains to choose from (along with their services):
        """
        for domain in domains:
            domain_content += f"{domain.name}: {domain.description}\n"
            for service in domain.services:
                domain_content += f"{service.name}: {service.description}\n"
        domain_content += "Please only respond with the domain name."

        messages = [
            Message(role="system", content=self.system_prompt + request),
            Message(role="context_ai", content=domain_content),
        ]

        if request_body is None:
            request_body = BaseRequestBody(
                messages=messages,
            )
        else:
            request_body.messages.extend(messages)

        # get domain from openai
        response = openai.ChatCompletion.create(
            **request_body.dict(),
        )

        assert response.choices[0].text in [
            domain.name for domain in domains
        ], f"Domain not found: {response.choices[0].text}"

        domain = response.choices[0].text

        request_body.messages.append(Message(role="assistant", content=domain))

        return domain, request_body

    def _get_states(self) -> List[State]:
        """Get a list of states"""

        url = f"{self.home_assistant_url}/api/states"
        response = requests.get(url, headers=self.home_assistant_headers)
        unparsed_states = response.json()
        states_response: List[State] = []
        for d in unparsed_states:
            state = State(**d)
            states_response.append(state)
        return states_response

    def _get_function_from_name(self, name: str) -> Function:
        """Get a function from its name"""

        functions = self._get_functions()
        for function in functions:
            if function.name == name:
                return function

    def _get_functions(self, domain: str = "") -> List[Function]:
        """Get a list of functions"""

        # get services
        services = self._get_domains()

        # generate functions file
        functions_file = self._generate_functions_file(services)
        logger.debug(f"functions_file: {functions_file}")

        # get functions
        function_dicts = json.load(open("functions.json", "r"))
        if domain:
            functions = [
                # turn dict into Function object only if dict['name'] starts with domain
                Function(**function_dict)
                for function_dict in function_dicts
                if function_dict["name"].startswith(domain)
            ]
        else:
            functions = [
                Function(**function_dict) for function_dict in function_dicts
            ]  # turn dict into Function object
        logger.debug(f"functions: {functions}")

        return functions

    def _get_domains(self, domain: str = "") -> List[Domain]:
        """Get services from home assistant API"""

        url = f"{self.home_assistant_url}/api/services"
        response = requests.get(url)
        unparsed_services = response.json()
        services_response: List[Domain] = []
        for d in unparsed_services:
            domain = Domain(**d)
            services_response.append(domain)

        if domain:
            return [domain for domain in services_response if domain.name == domain]
        return services_response

    @staticmethod
    def _generate_functions_file(services_response: List[Domain]):
        """Generate functions file using jinja2 template"""

        # get template
        template = Template(open("functions.jinja2", "r").read())

        # generate functions file
        functions_file = template.render(domains=services_response)
        with open("functions.json", "w") as f:
            f.write(functions_file)
            logger.debug(f"functions_file: {functions_file}")

        return functions_file

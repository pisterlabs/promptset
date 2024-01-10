import dataclasses
import json

from openai import OpenAI


from app.sources.amocrm.db import SearchModeSettings
from app.utils import err, misc
from app.utils.db import MethodResponse, Message
from app.working_modes.common import Mode
from app.working_modes.qualification_mode import QualificationMode
import pandas as pd


@dataclasses.dataclass
class SearchMode:
    s_m_data: SearchModeSettings

    @staticmethod
    def get_function(filename):
        df = pd.read_excel(filename)
        list_of_arrays = df.to_dict(orient='records')
        if len(list_of_arrays) == 0:
            return None

        result = {}
        for d in list_of_arrays:
            for k, v in d.items():
                if k not in result.keys():
                    if str(v).isdigit():
                        result[k] = {"type": "integer"}
                        continue
                    result[k] = {"type": "string", 'enum': []}
                if result[k]['type'] == 'string' and v not in result[k]['enum']:
                    result[k]['enum'].append(v)
        response = [{
            "name": "Function",
            "description": "Get flat request",
            "parameters": {
                "type": "object",
                "properties": result,
                'required': list(result.keys())
            }
        }]
        return response

    @staticmethod
    def get_keywords_values(message, func, gpt_key):
        messages = [
            {'role': 'system', 'content': 'Give answer:'},
            {"role": "user",
             "content": message}]

        client = OpenAI(api_key=gpt_key)
        response = client.chat.completions.create(model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=func,
        function_call="auto")
        print(response)
        print('eys')
        response_message = response.choices[0].message
        if response_message.get("function_call"):
            function_args = json.loads(response_message.function_call.arguments)
            return {'is_ok': True, 'args': function_args}
        else:
            return {'is_ok': False, 'args': {}}

    @staticmethod
    def find_from_database(filename, params, rules):
        df = pd.read_excel(filename)
        list_of_arrays = df.to_dict(orient='records')
        responses = []
        to_view = []
        print(rules)
        for d in list_of_arrays:
            approved = True
            for k, v in d.items():
                try:
                    if rules[k] == '=' and v != params[k]:
                        approved = False
                    elif rules[k] == '>=' and not (int(v) >= int(params[k])):
                        """Введенное значение должно быть больше или равно заданому"""
                        approved = False
                    elif rules[k] == '<=' and not (int(v) <= int(params[k])):
                        approved = False
                    elif rules[k] == '>' and not (int(v) > int(params[k])):
                        approved = False
                    elif rules[k] == '<' and not (int(v) < int(params[k])):
                        approved = False
                    elif rules[k] == '!' and not (k in to_view):
                        to_view.append(k)
                except Exception as e:
                    approved = False
                if not approved:
                    break
            if approved:
                responses.append(d)
        return responses, to_view

    @staticmethod
    def prepare_to_answer(choices, to_view, view_rule, results_count=1):
        resp = ""
        print(to_view)
        for i, choice in enumerate(choices[:results_count]):
            print(choice)
            rule = view_rule
            for v in choices[0].keys():
                rule = rule.replace("{" + str(v) + "}", str(choice[v]))
            resp += f'\n{rule}'
        return resp

    def execute(self, message, gpt_key) -> MethodResponse:
        filename = misc.download_file(self.s_m_data.database_link)
        function = SearchMode.get_function(filename)
        values = SearchMode.get_keywords_values(message, function, gpt_key)
        if not values['is_ok']:
            return MethodResponse(data=[Message("Ошибка детектинга")], all_is_ok=True, errors=set())
        choices, to_view = SearchMode.find_from_database(filename, values['args'], self.s_m_data.search_rules)

        if len(choices) == 0:
            return MethodResponse(data=[Message("В бд нет инфы")], all_is_ok=True, errors=set())

        prepared_message = SearchMode.prepare_to_answer(choices, to_view, self.s_m_data.view_rule,
                                                        self.s_m_data.results_count)

        return MethodResponse(data=[Message(prepared_message)], all_is_ok=True, errors=set())

        # Read file
        # Get function to analyze
        # Get params from openai
        # Find responses from database
        # Prepare answer to send

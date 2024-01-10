import json
import openai


class AuthenticationException(Exception):
    pass


class UnknownException(Exception):
    pass


class GptClient:
    def __init__(self, key, org):
        self._behavior = "You're a robot specialized in picking number from a list."
        self._random_function = {
            "name": "pick_random",
            "description": "Pick a random number from a list of numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chosen_number": {
                        "type": "string",
                        "description": "The chosen number from the list of numbers. Only return one number, never a list.",
                    },

                },
                "required": ["chosen_number"],
            },
        }
        openai.api_key, openai.organization = key, org

    def chat(self, message):
        system_msg = [{"role": "system", "content": self._behavior}]
        user_assistant_msgs = [{"role": "user", "content": message}]

        msgs = system_msg + user_assistant_msgs
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=msgs,
            functions=[self._random_function],
            function_call={"name": "pick_random"},
        )

        return response["choices"][0].to_dict()


class GptPlayer:
    def __init__(self, client):
        self._client = client

    def get_next_move(self, board):
        message = f"Pick a number from this list of numbers: {board}".replace("[", "").replace("]", "")
        try:
            response = self._client.chat(message)
        except openai.error.AuthenticationError:
            raise AuthenticationException("Invalid API key or organization ID.")
        except Exception as error:
            raise UnknownException(error)

        return self._parse_response(response)

    def _parse_response(self, response):
        funcs = response["message"]["function_call"]["arguments"]
        resp_dict = json.loads(funcs)
        return int(resp_dict["chosen_number"])



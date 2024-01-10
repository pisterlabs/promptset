from functools import wraps
import openai

class AiTools:
    def __init__(self, api_key, model_id = "text-davinci-003", max_tokens = 200, temperature = 0.5):
        openai.api_key = api_key
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature

    def ai_function(self, params=[("value", "str")], return_type="str"):
        def decorator(fn):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                function = (
                    "def "
                    + fn.__name__
                    + "("
                    + ", ".join(
                        map(lambda param: str(param[0]) + ":" + str(param[1]), params)
                    )
                    + ") -> "
                    + return_type
                    + ':\n    """\n    '
                    + fn.__doc__
                    + '\n    """'
                )
                return_values = ""
                for i, param in enumerate(params):
                    return_values += "\n    " + params[i][0] + ": " + str(args[i])
                propmpt = (
                    "You are now the following python function:\n```python\n"
                    + function
                    + "\n```\nThis is the usere input:"
                    + return_values
                    + "\n Only respond with the value thet the function returns!"
                )
                response = openai.Completion.create(
                    model=self.model_id, prompt=propmpt, temperature=self.temperature, max_tokens=self.max_tokens
                )
                return response["choices"][0]["text"].lstrip("\n")  # type: ignore

            return wrapper

        return decorator

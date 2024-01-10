import json
import openai


def construct_with_gpt4(model, **kwargs):
    schema_obj = json.loads(model.schema_json())
    user_input_str = json.dumps(kwargs)

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[{"role": "user", "content": user_input_str}],
        functions=[{"name": "set_result", "parameters": schema_obj}],
        function_call={"name": "set_result"},
    )

    result_str = completion.choices[0].message.function_call.arguments
    result_dict = json.loads(result_str)
    return model.parse_obj(result_dict)

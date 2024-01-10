from openai_util.function_call.funcation_invoke import FUNCTIONS


def register(function_info):
    def decorator(function):
        FUNCTIONS[function.__name__] = {
            'method': function,
            'info': function_info
        }
        return function

    return decorator


@register({
    "name": "get_map_navigation",
    "description": "Get directions to destination",
    "parameters": {
        "type": "object",
        "properties": {
            "destination": {
                "type": "string",
                "description": "destination name"
            }
        },
        "required": ["destination"]
    },
})
def get_map_navigation(destination):
    return {
        "destination": destination,
        "way": "If you are in a nearby city, you can choose to take the high-speed rail to Hangzhou. There are several high-speed rail stations in Hangzhou, including Hangzhou East Railway Station, Hangzhou Railway Station and Hangzhou South Railway Station. After arriving at the high-speed rail station, you can go to your destination by subway, taxi or bus"
    }

# datetime_tool.py
# A langchain tool that returns current local date and time
#
import datetime
import json
from langchain.agents import Tool


def time():
    # Get the current time
    current_time = datetime.datetime.now()
    # Format the time as a string in a local format

    local_time = current_time.strftime("%I:%M %p")
    return local_time


def date():
    # Get the current time
    current_time = datetime.datetime.now()

    # Format the time as a string in a local format
    local_time = current_time.strftime("%A, %B %d, %Y")
    return local_time


def datetime_tool(request: str = None) -> str:
    '''
    returns currend date and time

    Args:
        request (str): optional.
            If specified contains a list of specific variable needed, e.g.

            {"specific_variables":["time"]}

    Returns:
        date and time as a JSON data structure, in the format:

        '{{"fulldate":"<fulldate>","date":"<date>","time":"<time>"}}'
    '''

    data = {
        'date': date(),
        'time': time()
    }

    response_as_json = json.dumps(data)
    return response_as_json


#
# instantiate the langchain tool.
# The tool description instructs the LLM to pass data using a JSON.
# Note the "{{" and "}}": this double quotation is needed to avoid a runt-time error triggered by the agent instatiation.
#
name = "date_time"
# response_format = '{{"fulldate":"<fulldate>","date":"<date>","time":"<time>"}}'
request_format = '{{"specific_variables":["variable_name"]}}'
response_format = '{{"date":"<date>","time":"<time>"}}'
description = f'''
helps to retrieve date and time.
Input should be an optional JSON in the following format: {request_format}
Output is a JSON in the following format: {response_format}'
'''

# create an instance of the custom langchain tool
Datetime = Tool(
    name=name,
    func=datetime_tool,
    description=description,
    return_direct=False
)


if __name__ == '__main__':
    print(datetime_tool('{"specific_variables":["date"]}'))
    # => {"date": "Tuesday, February 14, 2023", "time": "07:22 PM"}

    print(Datetime)
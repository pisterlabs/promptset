from decouple import config
import openai
import subprocess
import json

openai.api_key = config("API_KEY")

message_flow = [
    {
        "role": "system", "content": """
You are an coder who provides working python code.
When presented with a coding request, You should provide working code as an argument to the function.
Base the data solely on the information provided in the request.
"""}
]

print()
message_flow.append({"role": "user", "content": f"""Please create a pie chart for these pytest results, The pie chart should focus on the number of tests in each test file, include a title and legend. The decimal points are each 1 test executed, total these to get the number of tests. These dots may cover multiple lines for the same test file.
Run python -m pytest
============================= test session starts ==============================
platform linux -- Python 3.10.6, pytest-7.4.0, pluggy-1.2.0
rootdir: /home/runner/work/cribbage_scorer/cribbage_scorer
collected 1098 items

tests/play/play_scorer_exceptions_test.py ......                         [  0%]
tests/play/play_scorer_test.py ......................................... [  4%]
.                                                                        [  4%]
tests/show/show_scorer__impossible_score_test.py ....................... [  6%]
........................................................................ [ 13%]
........................................................................ [ 19%]
........................................................................ [ 26%]
........................................................................ [ 32%]
........................................................................ [ 39%]
........................................................................ [ 45%]
........................................................................ [ 52%]
........................................................................ [ 58%]
........................................................................ [ 65%]
........................................................................ [ 72%]
........................................................................ [ 78%]
........................................................................ [ 85%]
........................................................................ [ 91%]
.........................................                                [ 95%]
tests/show/show_scorer_exceptions_test.py ...                            [ 95%]
tests/show/show_scorer_test.py ......................................... [ 99%]
......                                                                   [100%]

============================= 1098 passed in 1.13s ============================="""})


functions = [
    {
        "name": "run_python_code",
        "description": "Runs the code provided by chatGPT",
        "parameters": {
            "type": "object",
            "properties": {
                "the_code": {
                    "type": "string",
                    "description": "The Python code to complete the required task. You must include double escaped newlines in the code arguments.",
                }
            },
            "required": ["the_code"],
        }
    }
]

print(message_flow)
print("\nCalculating the results...\n")

response = openai.ChatCompletion.create(
    model="gpt-4-0613",
    messages=message_flow,
    functions=functions,
    function_call={"name": "run_python_code"},
    temperature=0,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)
response_message = response["choices"][0]["message"]
results = ""
print(response_message)
if response_message["function_call"]["name"] == "run_python_code":
    print(response_message["function_call"]["arguments"])
    code_in_json = json.loads(response_message["function_call"]["arguments"])
    the_actual_code = code_in_json["the_code"]
else:
    print("Bad things have occured, head for the hills.")
    exit(1)

print(f"Running the following code:\n{the_actual_code}")
print()
results = subprocess.run(["python",
                          "-c", the_actual_code],
                         capture_output=True)
print(results.stdout.decode("utf-8"))
print(results.stderr.decode("utf-8"))



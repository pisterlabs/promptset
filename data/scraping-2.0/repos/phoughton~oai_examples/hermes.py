from decouple import config
import openai
from file_chooser import choose_a_file


openai.api_key = config("API_KEY")

message_flow = [
    {
        "role": "system", "content": """
Your are an Software development engineer in Test who will review and report the results of some tests.
You will provide an accurate summary of the test results provided.

1) Give give details of test results in plain English
If a test failed, provide details of the failure.
Keep your response short and impersonal
Use numbers provided by the user and do not calculate numbers, totals or percentages.

2) Where asked, provide the same information in JSON format.

3) Where asked provide the same information in XML format.

The format should be as follows:


# Test results Summary

A summary description of all test results. Including:
    - The number of tests executed based on the data provided by the user.
    - The total number of tests passed and the number of tests failed based on the data provided by the user.

## Executive Summary of the test results

The executive summary should be a summary of the test results in plain English with a high standard of accuracy and vocabulary.

## Detailed Test Results for all tests executed

In this section repeat the following descriptiom for each and every test file mentioned in the user provided results:

1. [TEST FILE NAME] Description of the test
    - One or two sentence explanation of the test in plain English
    - [NUMBER OF TESTS FAILED]


## The details of the tests that failed in plain English

In this section explain the details of the test failure in plain English.

## The JSON version of these test passes and failures:
[PLACE THE JSON TEST RESULTS HERE]


## The XML version of these test passes and failures:
[PLACE THE XML TEST RESULTS HERE]

"""
    }
]

# Ask the user to choose the test results file and then read the contents of that file into a variable called test_results
test_results = choose_a_file("input", ".txt", "Please choose the test results file you want to review by number:")
print()
message_flow.append({"role": "user", "content": f"My pytest results for the tests are delimited here with 3 backticks. ```{test_results}```\n"})

message_flow.append({"role": "assistant", "content": """
I have read the test results and I will now provide a summary of the test results, this will be in the form of a markdown file.
"""})

print(message_flow)
print("\nCalculating the test results...\n")

response = openai.ChatCompletion.create(
    model="gpt-4-0613",
    messages=message_flow,
    temperature=0,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)

print(response)
print()

# Write the test results to a file in markdown format
with open("output/test_results.md", "w") as f:
    f.write("# The test results are as follows:\n")
    f.write(response["choices"][0]["message"]["content"])

print("The test results have been written to output/test_results.md")


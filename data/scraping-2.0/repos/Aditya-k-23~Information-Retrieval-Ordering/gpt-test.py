import os
import openai
import json
from dotenv import load_dotenv

load_dotenv()

"""
Take in a file name and parse the file into a dictionary of queries and
responses.

The dictionary is structured as follows:
    document = {
        curQuery: {
            1: [response1, response2, ...],
            2: [response1, response2, ...]
        }
    }

# Full-text results are 1
# Neural results are 2

"""


def parse_file(file_name):
    with open(file_name, "r") as file:
        document = {}
        for line in file:
            # Start of Query
            if line.startswith("Query is: "):
                # Read query
                curQuery = line[10:]
                print(curQuery)

                # Add query to document dictionary
                document[curQuery] = {}

                # Initialize query for full-text and neural results
                document[curQuery][1] = []
                document[curQuery][2] = []
                resultIdx = 0

            # Start of Result
            elif line.startswith("->>>>>>>>>>>>>>>>>>>>"):
                resultIdx += 1
                # Initialize 10 empty strings for each response
                for i in range(0, 50):
                    document[curQuery][resultIdx].append("")
                responseIdx = 0

            # End of result
            elif line.startswith("-" * 80):
                responseIdx += 1

            else:
                # Add response to document dictionary
                document[curQuery][resultIdx][responseIdx] += line
    return document


"""
Send a query to the GPT-3 API and return the response.
"""


def query_gpt(file_name):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    context = "You are a helpful assistant."
    prompt = "Given a JSON file that contains an instruction and fifty potential responses, please rank the quality of these responses, and return the index of the top 10 ranked responses, no additional explanation is needed."

    # Convert document to JSON
    document = parse_file(file_name)
    jsonDocument = json.dumps(document, indent=4)
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": context},
            {
                "role": "user",
                "content": prompt + jsonDocument,
            },
        ],
    )
    print(completion.choices[0].message)


########### TESTS ###########
# document = parse_file("./Queries/Emaar.txt")
query_gpt("./Queries/Emaar.txt")

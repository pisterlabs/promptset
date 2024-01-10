import unittest

from researchrobot.cmech.assistant import *

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_job",
            "description": "Search for a job experience by title and description.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The title of the job experience to search for. This should be a\nstring representing the job title, such as 'Software Engineer' or 'Data Analyst'."
                    },
                    "description": {
                        "type": "string",
                        "description": "The description of the job experience to search for. This should\nbe a detailed string describing the job role, responsibilities, or\nany specific details relevant to the search."
                    }
                }
            },
            "result": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The title of the experience."
                    },
                    "description": {
                        "type": "string",
                        "description": "The description of the experience."
                    },
                    "soc_codes": {
                        "type": "string",
                        "description": "The SOC codes of the experience."
                    }
                }
            }
        }
    }
]


class MyTestCase(unittest.TestCase):
    def test_something(self):
        import json

        from researchrobot.openai_tools.completions import openai_one_completion

        ex = Experience(title="Software Engineer", description="I am a software engineer.")

        messages = [
            {'role': 'system', 'content': 'We are testing the function calling features of the API.'},
            {'role': 'user',
             'content': "I'm going to give you an Experience object, and I want you to call the appropriate function with it."},
            {'role': 'user', 'content': json.dumps(ex.model_dump())}
        ]

        r = openai_one_completion(messages, tools=Functions.tools(), return_response=True)

        print(json.dumps(json.loads(r.model_dump_json()), indent=2))


if __name__ == '__main__':
    unittest.main()

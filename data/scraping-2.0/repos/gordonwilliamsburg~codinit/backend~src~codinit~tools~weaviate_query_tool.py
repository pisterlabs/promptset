import json
from typing import Optional

import weaviate
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools.base import BaseTool
from pydantic import ConfigDict


class WeaviateGraphQLTool(BaseTool):
    """Base tool for querying a GraphQL API."""

    name = "query_weaviate_graphql"
    description = """\
    Input to this tool is a similarity search query to the Weaviate GraphQL API, based on the provided schema, output is a result from the API.
    The input is a valid GraphQL query that can be directly used as input to a query function. No markdown backticks.
    If the query is not correct, an error message will be returned.
    If an error is returned with 'Bad request' in it, rewrite the query and try again.
    If an error is returned with 'Unauthorized' in it, do not try again, but tell the user to change their authentication.

    Use similarity search from weaviate, called using 'nearText', to search for few of the nearest entities regarding your query.
    Always query multiple elements! You might not get the exact result you need if you query only 1 element.
    Here are some examples for examples:

    Example input to get the top 6 imports related to langchain agents:
    {{
        Get {{
            Import(
            nearText: {{
                concepts: ["langchain agent"]
            }},
            limit: 3
            ) {{
            name
            parameters
            belongsToFile {{
                ... on File {{
                    name
                    }}
                }}
            }}
        }}
    }}

    It is always a good idea to set the limit to at least 3!! Also, your first choice is to look at imports!

    """  # noqa: E501
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(
        self,
        query: str,
        client: weaviate.Client,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        # language model sometimes returns the query inside of backticks
        query = query.replace("```", "")
        result = client.query.raw(query)
        return json.dumps(result, indent=2)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Graphql tool asynchronously."""
        raise NotImplementedError("WeaviateGraphQLTool does not support async")


weaviate_schema = """
schema = {
    "classes": [
        {
            "class": "Repository",
            "description": "A code repository",
            "properties": [
                {
                    "name": "name",
                    "dataType": ["text"],
                    "description": "name of the repository",
                },
                {
                    "name": "link",
                    "dataType": ["text"],
                    "description": "URL link to the remote repository",
                },
                {
                    "name": "hasFile",
                    "dataType": ["File"],
                    "description": "Files contained in the repository",
                },
            ],
        },
        {
            "class": "File",
            "description": "A file in a repository",
            "properties": [
                {
                    "name": "name",
                    "dataType": ["text"],
                    "description": "Name of the file",
                },
                {
                    "name": "link",
                    "dataType": ["text"],
                    "description": "Full link of the file in the remote repository",
                },
                {
                    "name": "hasImport",
                    "dataType": ["Import"],
                    "description": "Imports in the file",
                },
                {
                    "name": "hasClass",
                    "dataType": ["Class"],
                    "description": "Classes defined in the file",
                },
                {
                    "name": "hasFunction",
                    "dataType": ["Function"],
                    "description": "Functions defined in the file",
                },
            ],
        },
        {
            "class": "Import",
            "description": "An import in a file",
            "properties": [
                {
                    "name": "name",
                    "dataType": ["text"],
                    "description": "Name of the import",
                },
                {
                    "name": "belongsToFile",
                    "dataType": ["File"],
                    "description": "File the import is located in",
                },
            ],
        },
        {
            "class": "Class",
            "description": "A class in a file",
            "properties": [
                {
                    "name": "name",
                    "dataType": ["text"],
                    "description": "Name of the class",
                },
                {"name": "attributes", "dataType": ["text[]"]},
                {
                    "name": "hasFunction",
                    "dataType": ["Function"],
                    "description": "Functions defined in the class",
                },
                {
                    "name": "belongsToFile",
                    "dataType": ["File"],
                    "description": "File the class is defined in",
                },
            ],
        },
        {
            "class": "Function",
            "description": "A function in a file or class",
            "properties": [
                {
                    "name": "name",
                    "dataType": ["text"],
                    "description": "Name of the function",
                },
                {
                    "name": "code",
                    "dataType": ["text"],
                    "description": "Code body of the function",
                },
                {
                    "name": "parameters",
                    "dataType": ["text[]"],
                    "description": "Parameters of the function",
                },
                {
                    "name": "variables",
                    "dataType": ["text[]"],
                    "description": "Variables sued in the function",
                },
                {
                    "name": "return_value",
                    "dataType": ["text[]"],
                    "description": "Return values of the function",
                },
                {
                    "name": "belongsToFile",
                    "dataType": ["File"],
                    "description": "File the function is defined in",
                },
                {
                    "name": "belongsToClass",
                    "dataType": ["Class"],
                    "description": "Class the function is part of",
                },
            ],
        },
    ]
}
"""


graphql_fields = """Repository {
    name
    link
    File {
      name
      link
      Import {
        name
      }
      Class {
        name
        attributes
        Function {
          name
          code
          parameters
          variables
          return_value
        }
      }
      Function {
        name
        code
        parameters
        variables
        return_value
      }
    }
  }

"""

"""
Example Input to query the entity Class which has name LLMMathChain on the property called attributes:
    {{
        Get {{
            Class (where: {{ path: ["name"], operator: Equal, valueString: "LLMMathChain"}}) {{
                attributes
            }}
        }}
    }}

    Example Input to query the entity Function which has name initialize_agent on the property called parameters:
    {{
        Get {{
            Function (where: {{ path: ["name"], operator: Equal, valueString: "initialize_agent"}}) {{
                parameters
            }}
        }}
    }}

    Example Input to query the entity File which has name chat.py on the property hasImport and gives the names of the entity Import:
    {{
        Get {{
            File (where: {{ path: ["name"], operator: Equal, valueString: "chat.py" }}) {{
                hasImport {{
                    ... on Import {{
                        name
                    }}
                }}
            }}
        }}
    }}

    Example input to query the full import of the partial import BaseSingleActionAgent:
    {{
        Get {{
            Import (where: {{ path: ["name"], operator: Like, valueString: "%BaseSingleActionAgent%" }}) {{
                name
            }}
        }}
    }}

        Example input to get the top 5 objects of the entity class which are related to the concept of neural networks:
    {{
        Get {{
            Class(
            nearText: {{
                concepts: ["neural networks"]
            }},
            limit: 5
            ) {{
            name
            attributes
            belongsToFile {{
                ... on File {{
                    name
                    }}
                }}
            hasFunction {{
                ... on Function {{
                    name
                    }}
                }}
            }}
        }}
    }}

    Example input to get the top 3 Functions which are related to document extraction and their parameters:
    {{
        Get {{
            Function(
            nearText: {{
                concepts: ["Document extraction"]
            }},
            limit: 3
            ) {{
            name
            parameters
            belongsToFile {{
                ... on File {{
                    name
                    }}
            }}
            belongsToClass {{
                ... on Class {{
                    name
                    }}
                }}
            }}
        }}
    }}

"""

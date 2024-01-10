from instructor import OpenAISchema
from pydantic import Field


class Add2Numbers(OpenAISchema):
    """
    This function adds two numbers.
    """
    number1: int = Field(..., description="First number.")
    number2: int = Field(..., description="Second number.")



if __name__ == '__main__':
    import json

    openai_schema = Add2Numbers.openai_schema
    if '$defs' in openai_schema['parameters']:
        defs = openai_schema['parameters']['$defs']
        del openai_schema['parameters']['$defs']
    schema = {
        "openapi": "3.1.0",
        "info": {
            "title": "Query GA4 Data",
            "description": "Google Analytics 4 API",
            "version": "v1.0.0"
        },
        "servers": [
            {
                "url": ""  # enter your url here
            }
        ],
        "paths": {
            "/": {
                "post": {
                    "description": openai_schema['description'],
                    "operationId": "runReport",
                    "parameters": [],
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/RunReportParams"
                                }
                            }
                        },
                        "required": True,
                    },
                    "deprecated": False,
                    "security": [
                        {
                            "apiKey": []
                        }
                    ]
                }
            },
        },
        "components": {
            "schemas": {
                "RunReportParams": openai_schema['parameters'],
                **defs,
            },
            "securitySchemes": {
              "apiKey": {
                "type": "apiKey"
              }
            }
        },
    }
    print(json.dumps(schema, indent=2).replace("#/$defs/", "#/components/schemas/"))

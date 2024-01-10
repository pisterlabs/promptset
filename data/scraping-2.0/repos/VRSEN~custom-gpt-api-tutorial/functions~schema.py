from pydantic import Field
from typing import List, Optional, Literal
from instructor import OpenAISchema


class OrderBySchema(OpenAISchema):
    """
    Represents an order by condition for the GA4 query.
    """
    dimension_name: Optional[str] = Field(..., description="Dimension name to order by. Can either be a metric or a dimension.")
    metric_name: Optional[str] = Field(..., description="Metric name to order by. Can either be a metric or a dimension.")
    desc: bool = Field(True, description="Whether to order by descending or ascending.")

class DateRangeSchema(OpenAISchema):
    """
    Represents a date range for the GA4 query.
    """
    start_date: str = Field(..., description="Start date of the query.")
    end_date: str = Field(..., description="End date of the query.")

class MetricSchema(OpenAISchema):
    """
    Represents a metric for the GA4 query.
    """
    name: str = Field(..., description="Name of the metric.")

class DimensionSchema(OpenAISchema):
    """
    Represents a dimension for the GA4 query.
    """
    name: str = Field(..., description="Name of the dimension.")

class GA4QueryParams(OpenAISchema):
    """
    Parameters for querying the Google Analytics 4 API runReport endpoint.
    """
    date_ranges: List[DateRangeSchema] = Field(..., description="List of date ranges to query.")
    metrics: List[MetricSchema] = Field(..., description="List of metric names to query.")
    dimensions: Optional[List[DimensionSchema]] = Field([], description="List of dimension names to query.")
    order_bys: Optional[List[OrderBySchema]] = Field([], description="List of order bys to query.")
    limit: int = Field(5, description="Limit of the query. Defaults to 5.")


if __name__ == '__main__':
    import json

    openai_schema = GA4QueryParams.openai_schema
    defs = {}
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

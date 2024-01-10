import openai
from enum import Enum
from typing import List, Dict, Any, Tuple
from src.data.scores import Score, ScoreNames
from pydantic import BaseModel

from src.llm.utils import unpack_function_call_arguments


class ScoreConfig(BaseModel):
    name: ScoreNames
    range_min: int
    range_middle: int
    range_max: int
    range_min_description: str
    range_middle_description: str
    range_max_description: str
    var_name: str = ""

    def model_post_init(self, __context: Any) -> None:
        if self.var_name == "":
            self.var_name: str = self.name.value.lower().replace(" ", "_")
        return super().model_post_init(__context)


class ScoreType(Enum):
    SATISFACTION = ScoreConfig(
        name=ScoreNames.SATISFACTION,
        range_min=0,
        range_middle=50,
        range_max=100,
        range_min_description="very negative",
        range_middle_description="neutral",
        range_max_description="very positive",
    )
    SPECIFICITY = ScoreConfig(
        name=ScoreNames.SPECIFICITY,
        range_min=0,
        range_middle=50,
        range_max=100,
        range_min_description="not specific (lacking constructive detail)",
        range_middle_description="somewhat specific",
        range_max_description="very specific (highly constructive feedback)",
    )
    BUSINESS_IMPACT = ScoreConfig(
        name=ScoreNames.BUSINESS_IMPACT,
        range_min=0,
        range_middle=50,
        range_max=100,
        range_min_description="not impactful on business outcomes",
        range_middle_description="somewhat important",
        range_max_description="high severity (very impactful on business outcomes)",
    )


def generate_score_properties_and_required(
    score_types: List[ScoreType],
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """
    Generates the properties and required fields for the JSON schema for the OpenAI Functions API.

    Specifically, it does this for the score types provided.
    """
    properties: Dict[str, Dict[str, Any]] = {}
    required: List[str] = []

    for score_type in score_types:
        properties.update(
            {
                f"{score_type.value.var_name}_score": {
                    "type": "integer",
                    "description": f"The customer's {score_type.value.name} Score ONLY about what is mentioned in THAT observation. It is a relative score from {score_type.value.range_min} to {score_type.value.range_max} where {score_type.value.range_min} is {score_type.value.range_min_description}, {score_type.value.range_middle} is {score_type.value.range_middle_description}, and {score_type.value.range_max} is {score_type.value.range_max_description}.",
                    "minimum": score_type.value.range_min,
                    "maximum": score_type.value.range_max,
                },
                f"{score_type.value.var_name}_explanation": {
                    "type": "string",
                    "description": "An explanation for why you decided on that score for that observation.",
                },
            }
        )
        required.append(f"{score_type.value.var_name}_score")
        required.append(f"{score_type.value.var_name}_explanation")

    return (properties, required)


def score_observation(
    observation: str, feedback_item: str, score_types: List[ScoreType]
) -> List[Score]:
    model_name = "gpt-3.5-turbo-0613"

    messages = [
        {
            "role": "system",
            "content": "You are an expert in customer service. Your task is to report a score.",
        },
        {
            "role": "user",
            "content": f"""Here is a customer's complete feedback:\n{feedback_item}\n\nFrom this feedback, we have the following observation:\n{observation}\n\nFrom that observation, report the customer's scores on a continuous scale.""",
        },
    ]

    properties, required = generate_score_properties_and_required(score_types)

    functions = [
        {
            "name": "report_scores",
            "description": "Used to report the requested scores.",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }
    ]

    function_call = {"name": "report_scores"}

    response = openai.ChatCompletion.create(  # type: ignore
        model=model_name,
        messages=messages,
        functions=functions,
        function_call=function_call,
    )

    result = unpack_function_call_arguments(response)  # type: ignore

    scores: List[Score] = []
    for score_type in score_types:
        score = Score(
            name=score_type.value.name,
            score=result[score_type.value.var_name + "_score"],
            explanation=result[score_type.value.var_name + "_explanation"],
        )
        scores.append(score)

    return scores

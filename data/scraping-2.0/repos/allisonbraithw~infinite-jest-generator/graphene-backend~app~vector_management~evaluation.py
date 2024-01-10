import os
import openai
import json
from llama_index import ServiceContext
from llama_index.evaluation import RelevancyEvaluator
from llama_index.llms import OpenAI

openai.organization = os.environ.get("OPENAI_ORG")
openai.api_key = os.environ.get("OPENAI_API_KEY")


class DescriptionEval():
    relevancy: float
    explanation: str

    def __init__(self, relevancy, explanation) -> None:
        self.relevancy = relevancy
        self.explanation = explanation


def evaluate_relevancy_llama_index(query: str, docs: list, response: str) -> bool:
    # Evaluate relevancy based on the llama index evaluator

    gpt4 = OpenAI(temperature=0, model="gpt-4")
    service_context_gpt4 = ServiceContext.from_defaults(llm=gpt4)
    evaluator = RelevancyEvaluator(service_context=service_context_gpt4)
    eval_source_result_full = evaluator.evaluate(
        query=query,
        response=response,
        contexts=docs
    )
    print(eval_source_result_full)
    return eval_source_result_full.passing


# Evaluate relevancy based on a direct call to openAI
def evaluate_relevancy_bespoke(character: str, response: str) -> DescriptionEval:
    # pass description to OpenAI & ask it to assess relevancy
    system_prompt = 'You are a helpful AI assistant that takes a physical description of a character and assesses\
        the quality of that description on specified dimensions. \
        You should return a JSON formatted object with the following fields:\
            ```\
                {"relevancy": 0.0,\
                 "explanation": "This description is not relevant because it describes the character\'s personality, not their physical appearance."\
                }\
            ```\
        where relevancy is a 2 decimal-place float between 0 and 1, with 0 being not relevant at all and 1 being very relevant.\
        Consider factors such as, does the text describe a character physically (relevant), does it include extraneous non-physical \
            details (irrelevant), does it include information about other characters (irrelevant) \
        '

    messages = [
        {"role": "system", "content": f"{system_prompt}"},
        {
            "role": "user",
            "content": f"{response}",
        },
    ]
    print("Sending prompt to OpenAI")
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0.1,
    )
    json_response = json.loads(response.choices[0].message.content)
    return DescriptionEval(relevancy=json_response["relevancy"], explanation=json_response["explanation"])

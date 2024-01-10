from time import perf_counter

from dotenv import load_dotenv
from transformers import pipeline

from func_ai.utils.llm_tools import OpenAIInterface
from func_ai.utils.openapi_function_parser import OpenAPISpecOpenAIWrapper


def get_highest_score_label(data):
    scores = data['scores']
    highest_score = max(scores)
    highest_score_index = scores.index(highest_score)
    return data['labels'][highest_score_index]


def test_pipeline():
    load_dotenv()
    pipe = pipeline(model="facebook/bart-large-mnli")
    _spec = OpenAPISpecOpenAIWrapper.from_url('http://petstore.swagger.io/v2/swagger.json',
                                              llm_interface=OpenAIInterface())
    t0 = perf_counter()
    _sum = _spec.operations_summary
    _scored_labels = pipe("I want to add a new pet to the store.",
                          # here we give the function descriptions as labels to the model as this seems to work better
                          candidate_labels=[fn for fn in _sum.values()],
                          )
    elapsed = 1000 * (perf_counter() - t0)
    print("Inference time: %d ms.", elapsed)
    # return the index of the highest scoring label
    _highest_score_label = get_highest_score_label(_scored_labels)
    print(next((k for k, v in _sum.items() if v == _highest_score_label), None))
    # _max = max(_scored_labels, key=lambda x: x["score"])

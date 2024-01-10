# TODO: [AIS-83] Build a response evaluator that can be used to evaluate the quality of responses
from llama_index.evaluation import ResponseEvaluator
from langchain.chat_models import ChatOpenAI
from llama_index import LLMPredictor, ServiceContext
from llama_index.evaluation import QueryResponseEvaluator

# build service context
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-4"))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
evaluator = ResponseEvaluator(service_context=service_context)


def evaluate_response_binary(response) -> bool:
    """
    Evaluate a response to a query by using a language learning model to extract keywords from the input text.
    
    Args:
        response: Response to evaluate.
        query: Query to evaluate the response against.
        index: Vector store index to use for retrieving documents.
        
    Returns:
        A boolean representing whether the response is relevant to the query.
    """
    evaluator = ResponseEvaluator(service_context=service_context)
    eval_result = evaluator.evaluate(response)
    print(str(eval_result))
    return eval_result


def query_response_evaluator(query, response):
    """
    Evaluate a response to a query by using a language learning model to extract keywords from the input text.
    
    Args:
        response: Response to evaluate.
        query: Query to evaluate the response against.
        index: Vector store index to use for retrieving documents.
        
    Returns:
        A boolean representing whether the response is relevant to the query.
    """
    evaluator = QueryResponseEvaluator(service_context=service_context)
    eval_result = evaluator.evaluate(response, query)
    print(str(eval_result))
    return eval_result


if __name__ == "__main__":
    evaluate_response_binary()
    query_response_evaluator()

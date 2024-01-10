from haystack import Pipeline
from function_call import OpenAIFunctionCall

def create_pipeline(retriever, API_KEY):
    p = Pipeline()
    p.add_node(component=retriever, name="retriever", inputs=["Query"])
    p.add_node(component=OpenAIFunctionCall(API_KEY), name="OpenAIFunctionCall", inputs=["retriever"])
    return p
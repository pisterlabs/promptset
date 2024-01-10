from typing import Any, Dict, List, Union
from copy import deepcopy

from langchain.chat_models import ChatOllama
from langchain.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate

from langchain.callbacks.base import BaseCallbackHandler

from langchain_core.outputs import LLMResult
from langchain.callbacks.utils import (
    BaseMetadataCallbackHandler,
    flatten_dict,
    hash_string,
    import_pandas,
    import_spacy,
    import_textstat,
)

from metrics import send

metrics = dict()

def run_question(model, question):
    uri='http://127.0.0.1:8888'

    class MyCustomHandlerOne(BaseCallbackHandler):

        @staticmethod
        def add_to_metric(value):
            global metrics
            key, val = list(value.items())[0]
            if key in metrics and isinstance(val, int):
                metrics[key] += val
            elif isinstance(val, list):
                if key in metrics:
                    metrics[key].extend(val)
                else:
                    metrics[key]=val
            else:
                metrics.update(value)

        def on_llm_start(
            self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
                ) -> Any:
            self.add_to_metric({'system':serialized.get('id')[-1]})
            self.add_to_metric({'model':serialized.get('kwargs').get('model')})
            
        def on_llm_error(
            self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
                ) -> Any:
            """Run when LLM errors."""

        def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
            for generations in response.generations:
                for generation in generations:
                    response = flatten_dict(generation.dict())
                    self.add_to_metric({"response":[response]})
                

    callbacks = [MyCustomHandlerOne()]

    llm = ChatOllama(base_url=uri, model=model, temperature=0.0, callbacks=callbacks)

    graph = Neo4jGraph(
        url="bolt://localhost:7687",
        username="",
        password=""
    )

    CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer questions about infrastructure objects.
Convert the user's question based on the schema.

Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

If no data is returned, do not attempt to answer the question.
Only respond to questions that require you to construct a Cypher statement.
Do not include any explanations or apologies in your responses.

Examples: Here are a few examples of generated Cypher statements for particular questions:
# How many pods have default namespace?
MATCH (n:KubernetesNamespace {{name:'default'}})-[:HAS_POD]->(m) RETURN count(m) as pods
# How many containers have sample namespace?
MATCH (n:KubernetesNamespace {{ name: 'sample' }})-[:HAS_POD]->(pod)-[:HAS_CONTAINER]->(containers) return count(containers) as containers

Schema: {schema}
Question: {question}
"""

    cypher_generation_prompt = PromptTemplate(
        template=CYPHER_GENERATION_TEMPLATE,
        input_variables=["schema", "question"],
    )

    cypher_chain = GraphCypherQAChain.from_llm(
        llm,
        graph=graph,
        cypher_prompt=cypher_generation_prompt,
        verbose=True
    )
    out = cypher_chain.run(question)
    print(out)
    return out


if __name__ == "__main__":
    models = ['openchat:7b-v3.5-q2_K','llama2:13b','codellama:7b-instruct','deepseek-coder:latest']
    questions = {"How many pods have default namespace?":[2, "two"],"How many containers have default namespace?":[4, 'four']}

    attempts = 3
    for model in models:
        print(f"Test model - {model}")
        correct_answer_counter = 0
        for attempt in range(attempts):
            for question in questions.keys():
                try:
                    gen_answer = run_question(model, question)
                    if any( str(correct_reply) in gen_answer.lower() for correct_reply in questions.get(question) ):
                        correct_answer_counter += 1
                        metrics['correct_answers']=correct_answer_counter
                    metrics['total_answers']=attempts + len(questions) + 1
                    send(metrics)
                    metrics.clear()
                except Exception as e:
                    metrics.clear()
                    print("Произошла ошибка:", e)

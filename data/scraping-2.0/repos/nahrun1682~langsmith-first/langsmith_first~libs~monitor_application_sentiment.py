from langsmith.evaluation import EvaluationResult, RunEvaluator
from langsmith.schemas import Example, Run

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

from langsmith import Client

class SentimentEvaluator(RunEvaluator):
    def __init__(self):
        prompt = """Is the predominant sentiment in the following statement positive, negative, or neutral?
---------
Statement: {input}
---------
Respond in one word: positive, negative, or neutral.
Sentiment:"""

        llm = ChatOpenAI(model=settings.model_name, temperature=0.0)
        self.chain = LLMChain.from_string(llm=llm, template=prompt)

    def evaluate_run(self, run: Run, example: Example) -> EvaluationResult:
        input_str = str(list(run.inputs.values())[0])
        prediction = self.chain.run(input_str)
        # Strip the prompt
        prediction = prediction.strip()
        score = {"positive": 1, "negative": -1, "neutral": 0}.get(prediction)
        return EvaluationResult(
            key="sentiment",
            value=prediction,
            score=score,
        )


if __name__ == "__main__":
    client = Client()
    evaluator = SentimentEvaluator()
    for run in client.list_runs(
        project_name=settings.project_name,
        execution_order=1, # Do not return child / nested runs
    ):
        client.evaluate_run(run, evaluator)
    
from trulens_eval import Feedback
from trulens_eval.feedback import OpenAI as fOpenAI

# import numpy as np

class TruLensMeasures:

    def __init__(self):

# Initialize provider class
        fopenai = fOpenAI()

        # Question/answer relevance between overall question and answer.
        self.f_qa_relevance = (
            Feedback(fopenai.relevance, name = "Answer Relevance")
            .on_input_output()
        )

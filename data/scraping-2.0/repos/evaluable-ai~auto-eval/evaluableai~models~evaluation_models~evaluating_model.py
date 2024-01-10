from evaluableai.models.evaluation_models.evaluating_model_name import EvaluatingModelName
from evaluableai.models.evaluation_models.openai import Openai
from evaluableai.models.model import Model


class EvaluatingModel(Model):
    def __init__(self, model_name, model_version, api_key_env):
        # Initialize instance variables
        if model_name == EvaluatingModelName.OPENAI:
            self.instance = Openai(model_version, api_key_env)
        else:
            raise ValueError("Invalid models name")

    @property
    def model_name(self):
        return self._model_name

    @property
    def api_key(self):
        return self._api_key

    @property
    def model_version(self):
        return self._model_version

    def run_evaluation(self, input_frame):
        return self.instance.run_evaluation(input_frame)

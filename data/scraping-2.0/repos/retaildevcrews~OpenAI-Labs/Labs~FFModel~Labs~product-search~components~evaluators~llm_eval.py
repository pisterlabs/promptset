# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from ffmodel.components.base import BaseSolutionComponent
from ffmodel.data_models.base import ExperimentDataModel
from ffmodel.utils.openai import (
    OpenAIConfig,
    RetryParameters,
    filter_completion_arguments,
    generate_completion,
    initialize_openai,
)


class Component(BaseSolutionComponent[ExperimentDataModel]):
    """
    The LLM evaluator class for evaluating the quality or relevance of the completions using a Large Language Model.
    It uses the completion API.
    Note: The range of completion values output by this evaluator is based on the instruction file provided to the model.

    Component Args:
        - api_key_config_name: name of the config value to pull the api key from, defaults to OPENAI_API_KEY
        - api_endpoint_config_name: name of the config value to pull the api endpoint from, defaults to OPENAI_ENDPOINT
        - engine: model to use

    Component Config supporting_data:
        - static_instr_file: Path to the text file containing the static instructions for the prompt.
        The file provide a prompt template of step-by-step instructions and one example to guide the llm to generate evaluation score and explanation.
        The template should contain the following three variables:"prompt", "expected_output", and "completion".

    In addition the following args from openAI are most common, but any openAI arg can be passed:
        - stop: list of stop tokens
        - temperature: temperature as a float
        - max_tokens: max number of tokens to return from the model
        - top_p: 1.0
    """

    def _post_init(self):
        static_instr_file = self.supporting_data.get("static_instr_file", None)
        if static_instr_file is None:
            raise ValueError("Argument 'static_instr_file' must be provided")

        self.static_instr_file = static_instr_file.file_path
        with open(self.static_instr_file) as f:
            self.static_instr = f.read()

        config_names = self.args.pop("config", {})
        self.openai_config = OpenAIConfig.from_dict(config_names)
        retry_params = self.args.pop("retry_params", {})
        self.retry_params = RetryParameters.from_dict(retry_params)

        self.filtered_kwargs = filter_completion_arguments(self.args)

        self.engine = self.args.pop("engine", "engine")

        self.call_openai_function = generate_completion

    def execute(self, data_model: ExperimentDataModel) -> ExperimentDataModel:
        initialize_openai(self.openai_config)

        prompt = data_model.request.user_nl
        expected_output = data_model.request.expected_output
        completions = data_model.model_output.completions

        results = {"score": [], "explanation": []}
        if all(x is None for x in completions):
            raise ValueError("No completions provided.")

        for completion in completions:
            score, explanation = self.llm_score(prompt, expected_output[0], completion)
            results["score"].append(score)
            results["explanation"].append(explanation)

        data_model.experiment_metrics[self.get_id()] = results

        return data_model

    def llm_score(self, user_prompt, expected_output: str, completion: str) -> float:
        """Calculate the llm score between two strings using the completion API.

        Returns a float between min and max as defined in the prompt's instructions.
        """
        eval_prompt = self.static_instr.format(
            prompt=user_prompt, expected_output=expected_output, completion=completion
        )

        response = self.call_openai_function(
            prompt=eval_prompt,
            retry_parameters=self.retry_params,
            **self.filtered_kwargs,
        )

        score, explanation = self.get_score_exp(response.choices[0].text)

        return score, explanation

    def get_score_exp(self, completion):
        """Get the score and explanation from the completion text.

        Returns the score (float) and the explanation (string).
        """
        completion = completion.split("\n")
        # initialize the score and explanation
        explanation = ""
        score = 0
        # go over each line and find the score and explanation
        for line in completion:
            if "SCORE:" in line:
                # get the score
                try:
                    score = float(line.split(":")[1].strip())
                except ValueError:
                    score = 0
            if "EXPLANATION:" in line:
                # get the explanation
                explanation = line.split(":")[1].strip()

        return score, explanation

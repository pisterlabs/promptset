from typing import Dict, Any, Optional

from pandas import DataFrame

from actableai.parameters.options import OptionsParameter
from actableai.parameters.parameters import Parameters
from actableai.tasks import TaskType
from actableai.tasks.base import AAITask


class AAITextExtractionTask(AAITask):
    @staticmethod
    def _get_text_extraction_model_parameters() -> OptionsParameter[Parameters]:
        from actableai.text_extraction.models.base import Model
        from actableai.text_extraction.models import model_dict

        available_models = [
            Model.openai,
        ]
        default_model = Model.openai

        options = {}
        for model in available_models:
            model_parameters = model_dict[model].get_parameters()
            options[model] = {
                "display_name": model_parameters.display_name,
                "value": model_parameters,
            }

        return OptionsParameter[Parameters](
            name="text_extraction_model",
            display_name="Text Extraction Model",
            description="Model used to extract information from text",
            is_multi=False,
            default=default_model,
            options=options,
        )

    @classmethod
    def get_parameters(cls) -> Parameters:
        parameters = [
            cls._get_text_extraction_model_parameters(),
        ]

        return Parameters(
            name="text_extraction_parameters",
            display_name="Text Extraction Parameters",
            parameters=parameters,
        )

    @AAITask.run_with_ray_remote(TaskType.TEXT_EXTRACTION)
    def run(
        self,
        df: DataFrame,
        document_name_column: str,
        text_column: str,
        openai_api_key: str,
        openai_rate_limit_per_minute: float = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        import json
        import time
        import openai
        from actableai.data_validation.base import CheckLevels
        from actableai.data_validation.params import TextExtractionDataValidator
        from actableai.text_extraction.models import model_dict

        start = time.time()

        parameters_validation = None
        parameters_definition = self.get_parameters()
        if parameters is None or len(parameters) <= 0:
            parameters = parameters_definition.get_default()
        else:
            (
                parameters_validation,
                parameters,
            ) = parameters_definition.validate_process_parameter(parameters)

        data_validation_results = []

        if parameters_validation is not None:
            data_validation_results += parameters_validation.to_check_results(
                name="Parameters"
            )

        failed_checks = [x for x in data_validation_results if x is not None]
        if CheckLevels.CRITICAL in [x.level for x in failed_checks]:
            return {
                "status": "FAILURE",
                "validations": [
                    {"name": x.name, "level": x.level, "message": x.message}
                    for x in failed_checks
                ],
                "runtime": time.time() - start,
                "data": {},
            }

        model_name, model_parameters = next(
            iter(parameters["text_extraction_model"].items())
        )

        if model_name == "openai":
            model_parameters["rate_limit_per_minute"] = openai_rate_limit_per_minute

        model_class = model_dict[model_name]
        model = model_class(
            parameters=model_parameters,
            process_parameters=False,
        )

        data_validation_results += TextExtractionDataValidator().validate(
            df=df,
            document_name_column=document_name_column,
            text_column=text_column,
            fields_to_extract=model_parameters["fields_to_extract"],
        )

        failed_checks = [x for x in data_validation_results if x is not None]
        if CheckLevels.CRITICAL in [x.level for x in failed_checks]:
            return {
                "status": "FAILURE",
                "validations": [
                    {"name": x.name, "level": x.level, "message": x.message}
                    for x in failed_checks
                ],
                "runtime": time.time() - start,
                "data": {},
            }

        openai.api_key = openai_api_key

        extracted_data = model.predict(data=df[text_column])

        def try_parse(data):
            try:
                return json.loads(data)
            except ValueError:
                return None

        df["extracted_data_raw"] = extracted_data
        df["extracted_data"] = df["extracted_data_raw"].apply(try_parse)

        return {
            "data": {
                "extracted_data": df[
                    [
                        document_name_column,
                        "extracted_data",
                        "extracted_data_raw",
                    ]
                ],
            },
            "status": "SUCCESS",
            "messenger": "",
            "runtime": time.time() - start,
            "validations": [
                {"name": x.name, "level": x.level, "message": x.message}
                for x in failed_checks
            ],
        }

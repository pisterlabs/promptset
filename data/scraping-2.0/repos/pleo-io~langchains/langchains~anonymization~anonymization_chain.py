from __future__ import annotations

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

from langchain.chains.base import Chain
import base64

from typing import Any, Dict, List, Optional, Callable
from pydantic import Extra
from langchain.chains.base import Chain
from langchain.callbacks.manager import RunManager, AsyncRunManager

from langchain.utils.input import get_colored_text

from Crypto.Cipher import AES
from Crypto.Util.Padding import pad


class AnonymizationChain(Chain):
    key: bytes = b""

    output_key: str = "anonymized"
    input_key: str = "text"

    analyzer: AnalyzerEngine = None
    language_models = {
        "en": "en_core_web_lg",
    }
    languages = set(language_models.keys())
    transformation: Optional[Callable] = None
    de_transformation: Optional[Callable] = None

    verbose: bool = False

    def __init__(
        self,
        key: bytes,
        languages: List[str] = ["en"],
        transformation: Optional[Callable] = None,
        de_transformation: Optional[Callable] = None,
        verbose: bool = False,
    ):
        super().__init__()
        self.language_models = {
            key: value
            for key, value in self.language_models.items()
            if key in languages
        }
        self.languages = set(languages)
        self.analyzer = self.setup_analyzer()
        self.transformation = transformation
        self.de_transformation = de_transformation
        self.verbose = verbose
        self.key = key

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def setup_analyzer(self):
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [
                {"lang_code": lang, "model_name": model}
                for (lang, model) in self.language_models.items()
            ],
        }
        provider = NlpEngineProvider(nlp_configuration=configuration)
        npl_engine = provider.create_engine()

        return AnalyzerEngine(nlp_engine=npl_engine, supported_languages=self.languages)

    def anonymize(
        self,
        input: str,
        language: str,
        previous_results: dict,
        run_manager: Optional[RunManager | AsyncRunManager] = None,
    ) -> tuple[str, dict]:
        values = {}

        mapped = (
            map(lambda x: x.strip("<").strip(">"), previous_results.keys())
            if self.de_transformation is None
            else map(self.de_transformation, previous_results.keys())
        )
        allow_list = list(mapped)

        detected_pii = self.analyzer.analyze(
            text=input,
            entities=[
                "PERSON",
                "LOCATION",
                "CREDIT_CARD",
                "PHONE_NUMBER",
                "MEDICAL_LICENSE",
                "IBAN_CODE",
                "EMAIL_ADDRESS",
            ],
            score_threshold=0.85,
            language=language,
            allow_list=allow_list,
        )
        detected_pii = sorted(detected_pii, key=lambda x: x.start)
        anonymized = list(input)
        offset = 0
        for pii in detected_pii:
            value = input[pii.start : pii.end]
            if value in previous_results:
                continue

            if self.transformation is not None:
                replacement = self.transformation(value)
                values[replacement] = value
            else:
                cipher = AES.new(self.key, AES.MODE_CBC, self.key)
                encoded = value.encode("utf-8")
                encrypted = cipher.encrypt(pad(encoded, AES.block_size))
                stringified = base64.encodebytes(encrypted).decode("utf-8").strip("\n")
                replacement = f"<{pii.entity_type}-{stringified}>"
                values[replacement] = value

            start_index = pii.start + offset
            end_index = pii.end + offset

            anonymized[start_index:end_index] = replacement
            offset += len(replacement) - len(value)

        stringified = "".join(anonymized)
        if run_manager:
            _colored_text = get_colored_text(stringified, "green")
            _text = "Prompt after anonymization:\n" + _colored_text
            run_manager.on_text(text=_text, end="\n", verbose=self.verbose)
        return (stringified, values)

    def run_anonymization(
        self,
        inputs,
        run_manager: Optional[RunManager | AsyncRunManager] = None,
    ):
        input = inputs["text"]
        anonymized_values: dict[str, str] = {}
        anonymized, values = self.anonymize(
            input,
            language="en",
            previous_results=anonymized_values,
            run_manager=run_manager,
        )

        if "en" in self.languages:
            self.languages.remove("en")
            anonymized_values.update(values)

        for language in self.languages:
            new_anonymized, new_values = self.anonymize(
                anonymized,
                language=language,
                previous_results=values,
            )
            anonymized = new_anonymized
            anonymized_values.update(new_values)

        return {self.output_key: anonymized}

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[RunManager] = None,
    ) -> Dict[str, str]:
        return self.run_anonymization(inputs, run_manager=run_manager)

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncRunManager] = None,
    ) -> Dict[str, str]:
        return self.run_anonymization(inputs, run_manager=run_manager)

    @property
    def _chain_type(self) -> str:
        return "anonymization_chain"

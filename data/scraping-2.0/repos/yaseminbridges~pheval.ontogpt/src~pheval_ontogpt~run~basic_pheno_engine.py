"""Reasoner engine."""
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import openai.error
from jinja2 import Template
from oaklib import get_adapter
from oaklib.datamodels.text_annotator import TextAnnotationConfiguration
from oaklib.interfaces import MappingProviderInterface, TextAnnotatorInterface
from ontogpt.engines.knowledge_engine import KnowledgeEngine
from phenopackets import Diagnosis, Phenopacket
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class DiagnosisPrediction(BaseModel):
    case_id: str
    validated_disease_ids: List[str] = None
    validated_disease_labels: List[str] = None
    validated_mondo_disease_ids: List[str] = None
    validated_mondo_disease_labels: List[str] = None
    predicted_disease_ids: List[str] = None
    predicted_disease_labels: List[str] = None
    matching_disease_ids: List[str] = None
    rank: Optional[int] = None
    model: Optional[str] = None
    prompt: Optional[str] = None


@dataclass
class PhenoEngine(KnowledgeEngine):
    model = None
    completion_length = 700
    _mondo: TextAnnotatorInterface = None

    @property
    def mondo(self):
        if not self._mondo:
            self._mondo = get_adapter("sqlite:obo:mondo")
        return self._mondo

    def predict(
        self,
        phenopacket: Phenopacket,
        template_path: Union[str, Path] = None,
        constrained_list: [str] = None,
    ) -> List[Diagnosis]:
        # if template_path is None:
        #     template_path = DEFAULT_PHENOPACKET_PROMPT
        if isinstance(template_path, Path):
            template_path = str(template_path)
        if isinstance(template_path, str):
            # create a Jinja2 template object
            with open(template_path) as file:
                template_txt = file.read()
                template = Template(template_txt)
        try:
            hpo_terms = [hpo_term.type.label for hpo_term in phenopacket.phenotypic_features]
            if constrained_list is None:
                prompt = template.render(
                    hpo_terms=hpo_terms,
                )
            else:
                prompt = template.render(hpo_terms=hpo_terms, constrained_list=constrained_list)
            payload = self.client.complete(prompt, max_tokens=self.completion_length)
            payload = payload.replace(",\n  }", "\n  }")
            payload = payload.replace('"}', "}")
            payload = payload.replace("},\n ]", "}\n ]")
            last_brace_index = payload.rfind("}")
            payload = payload[: last_brace_index + 1] + payload[last_brace_index + 1 :].lstrip(",")
            try:  # try load as JSON
                obj = json.loads(payload)
                # self.enhance_payload(obj)
                return obj
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding - trying again: {payload}")
                match = re.search(r"\[.*?\]", payload, re.DOTALL)
                if match:
                    if match.group() != payload:
                        try:
                            obj = json.loads(match.group())
                            # self.enhance_payload(obj)
                            return obj
                        except json.JSONDecodeError as e:
                            logger.error(f"Error decoding JSON: {e}")
                            logger.error(f"Payload: {payload}")
                            return []
                logger.error(f"Error decoding JSON: {e}")
                logger.error(f"Payload: {payload}")
            return []
        except openai.error.InvalidRequestError:
            return []

    def evaluate(self, phenopackets: List[Phenopacket]) -> List[DiagnosisPrediction]:
        mondo = self.mondo
        if not isinstance(mondo, MappingProviderInterface):
            raise TypeError("Mondo adapter must implement MappingProviderInterface")

        results = []
        for phenopacket in phenopackets:
            dp = DiagnosisPrediction(case_id=phenopacket["id"], model=self.model)
            validated_disease_ids = {disease["term"]["id"] for disease in phenopacket["diseases"]}
            dp.validated_disease_ids = list(validated_disease_ids)
            dp.validated_disease_labels = [
                disease["term"]["label"] for disease in phenopacket["diseases"]
            ]
            dp.validated_mondo_disease_ids = []
            dp.validated_mondo_disease_labels = []
            for disease_id in validated_disease_ids:
                mondo_id = mondo.normalize(disease_id, target_prefixes=["MONDO"])
                if mondo_id:
                    dp.validated_mondo_disease_ids.append(mondo_id)
                    dp.validated_mondo_disease_labels.append(mondo.label(mondo_id))
                else:
                    logger.warning(f"Could not normalize {disease_id} to MONDO")
            diagnoses = self.predict(phenopacket)
            dp.predicted_disease_ids = []
            dp.predicted_disease_labels = []
            dp.rank = 999
            for i, diagnosis in enumerate(diagnoses):
                predicted_disease_ids = diagnosis["disease_ids"]
                dp.predicted_disease_ids.append(";".join(predicted_disease_ids))
                dp.predicted_disease_labels.append(diagnosis["disease"])
                matches = set(dp.validated_mondo_disease_ids).intersection(predicted_disease_ids)
                if matches:
                    print("Found match at index", i)
                    dp.rank = i
                    dp.matching_disease_ids = list(matches)
                    break
            # print(dump_minimal_yaml(dp.dict()))
            results.append(dp)
        return results

    def enhance_payload(self, diagnoses: List[Diagnosis]) -> List[Diagnosis]:
        """Enhance payload with additional information."""
        mondo = self.mondo
        config = TextAnnotationConfiguration(matches_whole_text=True)
        if not isinstance(mondo, TextAnnotatorInterface):
            raise ValueError("Mondo adapter must implement TextAnnotatorInterface")
        if type(diagnoses) is list:
            pass
        if type(diagnoses) is dict:
            diagnoses = diagnoses[list(diagnoses.keys())[0]]
        for diagnosis in diagnoses:
            disease_label = diagnosis["disease_name"]
            anns = list(mondo.annotate_text(disease_label, config))
            # print(anns)
            diagnosis["disease_ids"] = list(set([ann.object_id for ann in anns]))
        return diagnoses

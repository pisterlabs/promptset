import concurrent.futures
import logging
from typing import Dict, List

import openai

from vibraniumdome_shields.settings_loader import settings
from vibraniumdome_shields.shields.input.captains_shield import CaptainsShield
from vibraniumdome_shields.shields.input.model_denial_of_service_shield import ModelDenialOfServiceShield
from vibraniumdome_shields.shields.input.no_ip_in_urls_shield import NoIPInURLsShield
from vibraniumdome_shields.shields.input.prompt_injection_transformer_shield import PromptInjectionTransformerShield
from vibraniumdome_shields.shields.input.prompt_safety_shield import PromptSafetyShield
from vibraniumdome_shields.shields.input.regex_shield import InputRegexShield
from vibraniumdome_shields.shields.input.semantic_similarity_shield import SemanticSimilarityShield
from vibraniumdome_shields.shields.input.sensitive_information_disclosoure_shield import SensitiveInformationDisclosureShieldInput
from vibraniumdome_shields.shields.model import LLMInteraction, Risk, ShieldDeflectionResult, ShieldsDeflectionResult, VibraniumShield
from vibraniumdome_shields.shields.output.arbitrary_images_shield import ArbitraryImagesShield
from vibraniumdome_shields.shields.output.canary_token_disclosoure_shield import CanaryTokenDisclosureShield
from vibraniumdome_shields.shields.output.refusal_shield import RefusalShield
from vibraniumdome_shields.shields.output.regex_shield import OutputRegexShield
from vibraniumdome_shields.shields.output.sensitive_information_disclosoure_shield import SensitiveInformationDisclosureShieldOutput
from vibraniumdome_shields.shields.output.whitelist_urls_shield import WhitelistURLsShield
from vibraniumdome_shields.vector_db.vector_db_service import VectorDBService


class VibraniumShieldsFactory:
    logger = logging.getLogger(__name__)

    _vector_db_service: VectorDBService
    _input_shields: dict
    _output_shields: dict

    def __init__(self, _vector_db_service: VectorDBService):
        if not _vector_db_service:
            raise ValueError("VibraniumShieldsFactory missed VectorDB")

        openai.api_key = settings.get("OPENAI_API_KEY")
        self._vector_db_service = _vector_db_service
        self._input_shields = {
            shield._shield_name: shield
            for shield in [
                SemanticSimilarityShield(
                    self._vector_db_service,
                    settings.get("vibraniumdome_shields.semantic_similarity.min_prompt_len"),
                    settings.get("vibraniumdome_shields.semantic_similarity.default_thresold"),
                ),
                InputRegexShield(),
                CaptainsShield(settings.get("OPENAI_API_KEY")),
                PromptInjectionTransformerShield(settings.get("vibraniumdome_shields.transformer_model_name")),
                PromptSafetyShield(),
                SensitiveInformationDisclosureShieldInput(),
                ModelDenialOfServiceShield(),
                NoIPInURLsShield(),
            ]
        }

        self._output_shields = {
            shield._shield_name: shield
            for shield in [
                OutputRegexShield(),
                RefusalShield(settings.get("vibraniumdome_shields.refusal_model_name")),
                CanaryTokenDisclosureShield(),
                SensitiveInformationDisclosureShieldOutput(),
                ArbitraryImagesShield(),
                WhitelistURLsShield(),
            ]
        }

    def _create_shields_according_to_policy(self, policy, tag, shields) -> list:
        self.logger.debug("current tag: %s, policy: %s", tag, policy)
        policy_shields: list = policy.get("content").get(tag)
        shields_list: list = []

        for shield_policy in policy_shields:
            shield = shields.get(shield_policy["type"])
            if shield:
                shields_list.append([shield, shield_policy["metadata"]])
            else:
                self.logger.error("skip policy shield as its name is not supported: %s", shield_policy["type"])
        return shields_list


class CaptainLLM:
    _logger = logging.getLogger(__name__)

    _vibraniumdome_shields_factory: VibraniumShieldsFactory

    def __init__(self, vibraniumdome_shields_factory: VibraniumShieldsFactory):
        if not vibraniumdome_shields_factory:
            raise ValueError("CaptainLLM missed VibraniumShieldsFactory")
        self._vibraniumdome_shields_factory = vibraniumdome_shields_factory

    def _execute_captains_strategy(
        self, llm_interaction: LLMInteraction, shields: list, shield_deflection_result: ShieldsDeflectionResult, policy: dict
    ) -> Dict[str, List[ShieldDeflectionResult]]:
        execution_mode_async = settings.get("vibraniumdome_shields.execution_mode_async", default=True, cast="@bool")
        scan_id = shield_deflection_result.scan_id

        def deflect_shield(tuple: [VibraniumShield, dict]) -> List[ShieldDeflectionResult]:
            try:
                shield, shield_policy_config = tuple
                self._logger.info("run shield: %s, with scan id=%s", shield.name, scan_id)
                return shield.name, shield.deflect(llm_interaction, shield_policy_config, scan_id, policy)
            except Exception:
                self._logger.exception("error while deflecting shield %s with scan_id=%s", shield.name, scan_id)

        if execution_mode_async:
            with concurrent.futures.ThreadPoolExecutor(max_workers=5, thread_name_prefix="CaptainLLM") as executor:
                shields_res = executor.map(deflect_shield, shields)
                results = dict(filter(lambda x: len(x[1]) > 0, shields_res))
        else:
            results = {k: v for a_shield in shields for k, v in [deflect_shield(a_shield)] if len(v) > 0}

        shield_deflection_result.results = results
        self._calculate_risk(shield_deflection_result, policy)
        return shield_deflection_result

    def _calculate_risk(self, shield_deflection_result: ShieldsDeflectionResult, policy: dict):
        calculated_risk = 0
        hrt = policy.get("content", {}).get("high_risk_threshold", settings.get("vibraniumdome_shields.high_risk_threshold"))
        lrt = policy.get("content", {}).get("low_risk_threshold", settings.get("vibraniumdome_shields.low_risk_threshold"))
        for shield_name, matches in shield_deflection_result.results.items():
            for match in matches:
                calculated_risk = max(calculated_risk, match.risk)
                if match.risk >= hrt:
                    shield_deflection_result.high_risk_shields.add(shield_name)

        self._logger.debug("calculated risk for scan_id=%s is %d", shield_deflection_result.scan_id, calculated_risk)
        shield_deflection_result.risk_factor = calculated_risk
        if calculated_risk >= hrt:
            shield_deflection_result.risk = Risk.HIGH
        elif calculated_risk < lrt:
            shield_deflection_result.risk = Risk.NONE
        else:
            shield_deflection_result.risk = Risk.LOW

    def deflect_shields(self, llm_interaction: LLMInteraction, policy: dict) -> ShieldsDeflectionResult:
        shield_deflection_result = self.deflect_incoming(llm_interaction, policy)
        shield_deflection_result.merge(self.deflect_outbound(llm_interaction, policy))
        self._calculate_risk(shield_deflection_result, policy)
        return shield_deflection_result

    def deflect_incoming(self, llm_interaction: LLMInteraction, policy: dict) -> ShieldsDeflectionResult:
        self._logger.info("llm_interaction id: %s, llm_message: %s", llm_interaction.get_id(), llm_interaction.get_last_message())
        shield_deflection_result = ShieldsDeflectionResult()
        shields: list = self._vibraniumdome_shields_factory._create_shields_according_to_policy(
            policy, "input_shields", self._vibraniumdome_shields_factory._input_shields
        )
        return self._execute_captains_strategy(llm_interaction, shields, shield_deflection_result, policy)

    def deflect_outbound(self, llm_interaction: LLMInteraction, policy: dict) -> ShieldsDeflectionResult:
        self._logger.info("llm_interaction id: %s, llm_message: %s", llm_interaction.get_id(), llm_interaction.get_last_message())
        shield_deflection_result = ShieldsDeflectionResult()
        shields: list = self._vibraniumdome_shields_factory._create_shields_according_to_policy(
            policy, "output_shields", self._vibraniumdome_shields_factory._output_shields
        )
        return self._execute_captains_strategy(llm_interaction, shields, shield_deflection_result, policy)

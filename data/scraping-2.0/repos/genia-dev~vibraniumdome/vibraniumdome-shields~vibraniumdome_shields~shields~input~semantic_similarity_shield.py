import logging
from typing import List, Optional, Tuple
from uuid import UUID

from langchain.docstore.document import Document

from vibraniumdome_shields.shields.model import LLMInteraction, ShieldDeflectionResult, VibraniumShield
from vibraniumdome_shields.vector_db.vector_db_service import VectorDBService


class SemanticSimilarityShieldDeflectionResult(ShieldDeflectionResult):
    text: str = ""
    metadata: Optional[dict] = {}
    distance: float = 0.0


class SemanticSimilarityShield(VibraniumShield):
    logger = logging.getLogger(__name__)
    _shield_name: str = "com.vibraniumdome.shield.input.semantic_similarity"
    _vector_db_service: VectorDBService
    _min_prompt_len: int
    _default_threshold: int

    def __init__(self, vector_db_service: VectorDBService, min_prompt_len: int, default_threshold: int):
        super().__init__(self._shield_name)
        self._vector_db_service = vector_db_service
        self._min_prompt_len = min_prompt_len
        self._default_threshold = default_threshold

    def deflect(self, llm_interaction: LLMInteraction, shield_policy_config: dict, scan_id: UUID, policy: dict) -> List[ShieldDeflectionResult]:
        threshold = shield_policy_config.get("threshold", 0.4)
        llm_message = llm_interaction.get_last_user_message_or_function_result()
        shield_matches = []

        # ignore short prompts as they are too flaky for semantic similarity
        if len(llm_message) >= self._min_prompt_len:
            try:
                matches: List[Tuple[Document, float]] = self._vector_db_service.query(llm_message, 3)
                existing_texts: set = set()

                for match in matches:
                    distance = match[1]
                    text = match[0].page_content
                    if distance < threshold and text not in existing_texts:
                        # with vector db a lower distance means higher vectors cosine similarity
                        shield_matches.append(SemanticSimilarityShieldDeflectionResult(text=text, metadata=match[0].metadata, distance=distance, risk=1))
                        # TODO: extract this logic to another strategy elsewhere
                        existing_texts.add(text)
            except Exception as err:
                self.logger.exception("Failed to perform vector shield, scan_id=%d", scan_id)
                raise err

        if len(shield_matches) == 0:
            shield_matches.append(SemanticSimilarityShieldDeflectionResult())

        return shield_matches

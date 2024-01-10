from api.application.handler.question_handler import QuestionHandler
from api.domain.use_case.question_use_cases import QuestionUseCases
from api.infrastructure.storage.repository.question_repository import QuestionRepository
from api.infrastructure.service.unique_id_generator import UniqueIdGeneratorService
from api.domain.factories import QuestionFactory
from api.infrastructure.artificial_intelligence.openai_provider import OpenAiProvider

QUESTION_REPOSITORY = QuestionRepository()
UNIQUE_ID_GENERATOR = UniqueIdGeneratorService()
QUESTION_FACTORY = QuestionFactory(uniqueIdGenerator=UNIQUE_ID_GENERATOR)
OPEN_AI_PROVIDER = OpenAiProvider()
QUESTION_USE_CASES = QuestionUseCases(
    repository=QUESTION_REPOSITORY,
    questionFactory=QUESTION_FACTORY,
    artificialIntelligence=OPEN_AI_PROVIDER,
)
QUESTION_HANDLER = QuestionHandler(question_use_cases=QUESTION_USE_CASES)

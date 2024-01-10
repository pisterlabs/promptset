import cachetools
from database import table_survey_record, table_survey
from agent import langchain_agent
import logging

survey_record_table = table_survey_record.SurveyRecordTable()
survey_table = table_survey.BusinessSurveyTable()

logger = logging.getLogger(__name__)

class PersistentTTLCache(cachetools.TTLCache):
    def __init__(self, maxsize: int, ttl: int):
        super().__init__(maxsize, ttl)

    def popitem(self):
        # Call the parent's popitem to get the evicted item
        record_id, agent = super().popitem()
        chat_history = agent.extract_chat_history()
        survey_record_table.update_chat_history(record_id=record_id, chat_history=chat_history)

        return record_id, agent


class ConversationManager:
    def __init__(
        self,
        cache_size: int,
        ttl: int,
    ) -> None:
        self.cache = PersistentTTLCache(maxsize=cache_size, ttl=ttl)

    def get_agent_from_record(self, record_id: str, survey_id: str) -> langchain_agent.LangChainAgent:
        if record_id in self.cache:
            return self.cache[record_id]
        # check db if there's existing history
        survey_record = survey_record_table.get_item(record_id=record_id)
        # if it does, create agent from history
        if survey_record is not None:
            chat_history = survey_record.chat_history
            if chat_history is not None:
                self.cache[record_id] = langchain_agent.LangChainAgent(conversation_history=chat_history)
                return self.cache[record_id]

        logger.info("record not found, creating a new one")
        # if not, get system prompt from survey table and create agent from there
        system_message, initial_message = survey_table.get_prompt_from_survey_id(survey_id=survey_id)
        self.cache[record_id] = langchain_agent.LangChainAgent(
            system_message=system_message,
            initial_message=initial_message,
        )
        return self.cache[record_id]

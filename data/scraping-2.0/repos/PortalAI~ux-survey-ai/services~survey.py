from fastapi_cognito import CognitoToken

from agent import langchain_agent
from config import settings
from database import table_survey, table_survey_record
from model.database_model import BusinessSurvey
from model.service import GetSurveyResponse
from services.auth import Auth

survey_table = table_survey.BusinessSurveyTable()
survey_record_table = table_survey_record.SurveyRecordTable()


class SurveyService:
    @staticmethod
    def get_survey(survey_id: str, auth: CognitoToken) -> GetSurveyResponse:
        survey = survey_table.get_item(survey_id)
        Auth.validate_permission(survey, auth)
        records = survey_record_table.list_survey_records(survey_id=survey_id)
        return GetSurveyResponse(**survey.model_dump(),
                                 chat_link=f"{settings.customer_chat_root_link}/survey/{survey.survey_id}",
                                 survey_records_count=len(records))

    @staticmethod
    def create_init_message(system_prompt: str) -> str:
        agent = langchain_agent.LangChainAgent(system_message=system_prompt)
        return agent.generate_response(system_prompt)

    @staticmethod
    def build_survey_response(survey: BusinessSurvey) -> GetSurveyResponse:
        records = survey_record_table.list_survey_records(survey_id=survey.survey_id)
        return GetSurveyResponse(**survey.model_dump(),
                                 survey_records_count=len(records),
                                 chat_link=f"{settings.customer_chat_root_link}/survey/{survey.survey_id}")

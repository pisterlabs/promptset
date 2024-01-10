from django.contrib.auth.models import AbstractBaseUser, AnonymousUser

from user_interface.managers import OpenAiApiManager
from user_interface.constants import INSTRUCTIONS_TYPES, USER_HISTORY_TYPES
from user_interface.models import ImprovementHistory


class ImproveTextManager:
    """Manager for improving text using OpenAI API."""

    def improve_text(self, text: str, improvement_type: str) -> str:
        """Improve the given text based on the specified improvement type."""
        openai_manager: OpenAiApiManager = OpenAiApiManager()
        instruction_prompt: str = [
            item for item in INSTRUCTIONS_TYPES if item[0] == improvement_type
        ][0][1]
        return openai_manager.send_api_request(text, instruction_prompt)

    def save_improvement_history(
        self,
        user: AbstractBaseUser | AnonymousUser,
        history_type: str,
        history_improvement_type: str,
        text: str,
        response_text: str,
    ) -> ImprovementHistory:
        text_type: str = [
            item for item in USER_HISTORY_TYPES if item[1] == history_type
        ][0][0]
        improvement_history = ImprovementHistory.objects.create(
            user=user,
            history_type=text_type,
            history_improvement_type=history_improvement_type,
            text=text,
            response_text=response_text,
        )

        return improvement_history

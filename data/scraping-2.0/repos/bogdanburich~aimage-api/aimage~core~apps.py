import openai
from django.apps import AppConfig
from django.conf import settings


class CoreConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'core'

    def ready(self) -> None:
        """App base settings."""
        openai.api_key = settings.OPENAI_API_KEY
        openai.organization = settings.OPENAI_ORG_KEY
        return super().ready()

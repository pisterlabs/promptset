import openai
from django.apps import AppConfig
from django.conf import settings


class ChatConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.chat"

    def ready(self):
        if settings.OPENAI_HTTP_PROXY_URL:
            openai.proxy = {"http": settings.OPENAI_HTTP_PROXY_URL, "https": settings.OPENAI_HTTPS_PROXY_URL}

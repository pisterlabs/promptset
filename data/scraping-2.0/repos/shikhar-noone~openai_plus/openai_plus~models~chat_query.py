from django.db import models
from openai_plus.models.base import AuthorTimeStampedModel


class ChatQuery(AuthorTimeStampedModel):
    query = models.TextField(default='')
    result = models.TextField(default='')
    language = models.TextField(default='')
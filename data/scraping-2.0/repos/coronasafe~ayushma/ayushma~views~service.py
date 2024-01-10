import time
from types import SimpleNamespace

import openai
from django.conf import settings
from django.http import StreamingHttpResponse
from drf_spectacular.utils import extend_schema, extend_schema_view, inline_serializer
from rest_framework import permissions, status
from rest_framework.decorators import action
from rest_framework.exceptions import ValidationError
from rest_framework.mixins import CreateModelMixin
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from rest_framework.serializers import CharField, IntegerField

from ayushma.models import APIKey, Chat, ChatMessage, Project
from ayushma.serializers import ChatDetailSerializer, ConverseSerializer
from ayushma.serializers.services import TempTokenSerializer
from ayushma.utils.converse import converse_api
from ayushma.utils.language_helpers import translate_text
from ayushma.utils.openaiapi import converse
from utils.views.base import BaseModelViewSet

from .chat import ChatViewSet


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


class APIKeyAuth(permissions.BasePermission):
    def has_permission(self, request, view):
        if request.headers.get("X-API-KEY"):
            api_key = request.headers.get("X-API-KEY")
            try:
                key = APIKey.objects.get(key=api_key)
                return True
            except APIKey.DoesNotExist:
                return False


class TempTokenViewSet(BaseModelViewSet, CreateModelMixin):
    serializer_class = TempTokenSerializer
    permission_classes = (APIKeyAuth,)
    lookup_field = "external_id"

    def perform_create(self, serializer):
        api_key = self.request.headers.get("X-API-KEY")
        key = APIKey.objects.get(key=api_key)

        serializer.save(api_key=key)

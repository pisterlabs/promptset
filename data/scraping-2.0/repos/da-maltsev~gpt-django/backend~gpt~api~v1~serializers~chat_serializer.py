from gpt.definitions import OPENAI_ROLES
from rest_framework import serializers


class ChatMessageSerializer(serializers.Serializer):
    role = serializers.ChoiceField(choices=OPENAI_ROLES, required=True)
    content = serializers.CharField(min_length=1, max_length=2000, required=True)


class ChatSerializer(serializers.Serializer):
    messages = serializers.ListField(child=ChatMessageSerializer(), required=True)

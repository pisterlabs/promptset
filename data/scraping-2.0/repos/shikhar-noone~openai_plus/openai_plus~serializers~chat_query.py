from rest_framework import serializers
from openai_plus.models import ChatQuery

class ChatQuerySerializer(serializers.ModelSerializer):

    class Meta:
        model = ChatQuery
        fields = "__all__"



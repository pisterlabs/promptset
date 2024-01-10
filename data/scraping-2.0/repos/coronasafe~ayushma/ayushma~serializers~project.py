from random import sample

from django.conf import settings
from openai import OpenAI
from rest_framework import serializers

from ayushma.models import Project


class ProjectSerializer(serializers.ModelSerializer):
    display_preset_questions = serializers.SerializerMethodField()

    def get_display_preset_questions(self, project_object):
        if project_object.preset_questions and len(project_object.preset_questions) > 4:
            return sample(project_object.preset_questions, 4)
        else:
            return project_object.preset_questions

    class Meta:
        model = Project
        fields = (
            "external_id",
            "title",
            "created_at",
            "modified_at",
            "description",
            "stt_engine",
            "model",
            "is_default",
            "display_preset_questions",
            "archived",
            "assistant_id",
        )
        read_only_fields = (
            "external_id",
            "created_at",
            "modified_at",
            "display_preset_questions",
        )


class ProjectUpdateSerializer(serializers.ModelSerializer):
    display_preset_questions = serializers.SerializerMethodField()
    key_set = serializers.SerializerMethodField()
    model = serializers.CharField(required=False)

    def get_display_preset_questions(self, project_object):
        if project_object.preset_questions and len(project_object.preset_questions) > 4:
            return sample(project_object.preset_questions, 4)
        else:
            return project_object.preset_questions

    class Meta:
        model = Project
        fields = ProjectSerializer.Meta.fields + (
            "prompt",
            "open_ai_key",
            "key_set",
            "preset_questions",
        )
        extra_kwargs = {
            "open_ai_key": {"write_only": True},
        }
        read_only_fields = ("key_set", "display_preset_questions")

    def update(self, instance, validated_data):
        if validated_data.get("is_default", True):
            Project.objects.all().update(is_default=False)

        project_id = self.context["view"].kwargs["external_id"]
        project = Project.objects.get(external_id=project_id)
        assistant_id = validated_data.get("assistant_id") or project.assistant_id

        if assistant_id:
            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            assistant = client.beta.assistants.retrieve(assistant_id)
            prompt = validated_data.pop("prompt", assistant.instructions)
            model = validated_data.pop("model", assistant.model)
            client.beta.assistants.update(
                assistant_id, instructions=prompt, model=model
            )

        return super().update(instance, validated_data)

    def get_key_set(self, obj):
        return obj.open_ai_key is not None

from django.conf import settings
from django_filters.rest_framework import DjangoFilterBackend
from openai import OpenAI
from rest_framework import filters, status
from rest_framework.decorators import action
from rest_framework.mixins import (
    CreateModelMixin,
    DestroyModelMixin,
    ListModelMixin,
    RetrieveModelMixin,
)
from rest_framework.permissions import IsAdminUser, IsAuthenticated
from rest_framework.response import Response

from ayushma.models import Project
from ayushma.serializers.project import ProjectSerializer, ProjectUpdateSerializer
from utils.views.base import BaseModelViewSet
from utils.views.mixins import PartialUpdateModelMixin


class ProjectViewSet(
    BaseModelViewSet,
    PartialUpdateModelMixin,
    ListModelMixin,
    RetrieveModelMixin,
    CreateModelMixin,
    DestroyModelMixin,
):
    queryset = Project.objects.all()
    filter_backends = (filters.SearchFilter, DjangoFilterBackend)
    search_fields = ("title",)
    filterset_fields = ("archived",)
    serializer_class = ProjectSerializer
    permission_classes = (IsAdminUser,)
    permission_action_classes = {
        "list": (IsAuthenticated(),),
        "retrieve": (IsAuthenticated(),),
    }
    lookup_field = "external_id"

    def destroy(self, request, *args, **kwargs):
        if self.request.user.is_staff:
            return super().destroy(request, *args, **kwargs)
        return Response(
            {"non_field_errors": "You do not have permission to delete this project"},
            status=400,
        )

    def get_serializer_class(self):
        if self.request.user.is_staff:
            return ProjectUpdateSerializer
        return super().get_serializer_class()

    def get_queryset(self):
        queryset = super().get_queryset()
        if self.action == "list":
            if not self.request.user.is_staff:
                queryset = self.queryset.filter(is_default=True)
        return queryset

    def perform_create(self, serializer):
        serializer.save(creator=self.request.user)

    def perform_destroy(self, instance):
        # delete namespaces from vectorDB
        try:
            settings.PINECONE_INDEX_INSTANCE.delete(
                namespace=str(instance.external_id),
                deleteAll=True,
            )
        except Exception as e:
            print(e)
            return Response(
                {
                    "non_field_errors": "Error deleting documents from vectorDB for this project"
                },
                status=400,
            )
        return super().perform_destroy(instance)

    @action(detail=True, methods=["post"])
    def create_assistant(self, request, *args, **kwarg):
        name = request.data.get("name")
        project: Project = Project.objects.get(external_id=kwarg["external_id"])

        # Since all the threads are attached to the api key, we use the env variable to avoid user confusion due to accidental key change
        client = OpenAI(api_key=settings.OPENAI_API_KEY)

        if project.assistant_id:
            return Response(
                {"non_field_errors": "Assistant already exists for this project"},
                status=400,
            )

        instructions = request.data.get("instructions")
        model = request.data.get("model")

        try:
            response = client.beta.assistants.create(
                name=name,
                instructions=instructions,
                model=model,
                tools=[{"type": "retrieval"}],
            )
            project.assistant_id = response.id
            project.save()
            return Response({"assistant_id": response.id})
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=True, methods=["get"])
    def list_assistants(self, *args, **kwarg):
        client = OpenAI(api_key=settings.OPENAI_API_KEY)

        assistants = client.beta.assistants.list(
            order="desc",
            limit="100",
        ).data

        return Response(
            [
                {
                    "id": assistant.id,
                    "name": assistant.name,
                    "instructions": assistant.instructions,
                    "model": assistant.model,
                }
                for assistant in assistants
            ]
        )

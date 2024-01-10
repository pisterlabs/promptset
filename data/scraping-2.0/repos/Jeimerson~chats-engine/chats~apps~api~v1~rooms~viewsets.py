from django.conf import settings
from django.db.models import Max
from django.utils import timezone
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework import filters, mixins, permissions, status
from rest_framework.decorators import action
from rest_framework.filters import OrderingFilter
from rest_framework.response import Response
from rest_framework.viewsets import GenericViewSet

from chats.apps.api.v1 import permissions as api_permissions
from chats.apps.api.v1.internal.rest_clients.openai_rest_client import OpenAIClient
from chats.apps.api.v1.msgs.serializers import ChatCompletionSerializer
from chats.apps.api.v1.rooms import filters as room_filters
from chats.apps.api.v1.rooms.serializers import (
    RoomMessageStatusSerializer,
    RoomSerializer,
    TransferRoomSerializer,
)
from chats.apps.dashboard.models import RoomMetrics
from chats.apps.rooms.models import Room
from chats.apps.rooms.views import (
    close_room,
    get_editable_custom_fields_room,
    update_custom_fields,
    update_flows_custom_fields,
    create_transfer_json,
    create_room_feedback_message,
)


class RoomViewset(
    mixins.ListModelMixin,
    mixins.RetrieveModelMixin,
    mixins.UpdateModelMixin,
    GenericViewSet,
):
    queryset = Room.objects.all()
    serializer_class = RoomSerializer
    filter_backends = [
        OrderingFilter,
        DjangoFilterBackend,
        filters.SearchFilter,
    ]
    filterset_class = room_filters.RoomFilter
    search_fields = ["contact__name", "urn"]
    ordering_fields = "__all__"
    ordering = ["user", "-last_interaction"]

    def get_permissions(self):
        permission_classes = [permissions.IsAuthenticated]
        if self.action != "list":
            permission_classes = (
                permissions.IsAuthenticated,
                api_permissions.IsQueueAgent,
            )
        return [permission() for permission in permission_classes]

    def get_queryset(self):
        if self.action != "list":
            self.filterset_class = None
        qs = super().get_queryset()
        return qs.annotate(last_interaction=Max("messages__created_on"))

    def get_serializer_class(self):
        if "update" in self.action:
            return TransferRoomSerializer
        return super().get_serializer_class()

    @action(
        detail=True,
        methods=[
            "PATCH",
        ],
        url_name="bulk_update_msgs",
        serializer_class=RoomMessageStatusSerializer,
    )
    def bulk_update_msgs(self, request, *args, **kwargs):
        room = self.get_object()
        if room.user is None:
            return Response(
                {"detail": "Can't mark queued rooms as read"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        serializer = RoomMessageStatusSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        serialized_data = serializer.validated_data

        message_filter = {"seen": not serialized_data.get("seen")}
        if request.data.get("messages", []):
            message_filter["pk__in"] = request.data.get("messages")

        room.messages.filter(**message_filter).update(
            modified_on=timezone.now(), seen=serialized_data.get("seen")
        )
        room.notify_user("update")
        return Response(
            {"detail": "All the given messages have been marked as read"},
            status=status.HTTP_200_OK,
        )

    @action(detail=True, methods=["PUT", "PATCH"], url_name="close")
    def close(
        self, request, *args, **kwargs
    ):  # TODO: Remove the body options on swagger as it won't use any
        """
        Close a room, setting the ended_at date and turning the is_active flag as false
        """
        # Add send room notification to the channels group
        instance = self.get_object()

        tags = request.data.get("tags", None)
        instance.close(tags, "agent")
        serialized_data = RoomSerializer(instance=instance)
        instance.notify_queue("close", callback=True)
        instance.notify_user("close")

        if not settings.ACTIVATE_CALC_METRICS:
            return Response(serialized_data.data, status=status.HTTP_200_OK)

        close_room(str(instance.pk))
        return Response(serialized_data.data, status=status.HTTP_200_OK)

    def perform_create(self, serializer):
        serializer.save()
        serializer.instance.notify_queue("create")

    def perform_update(self, serializer):
        # TODO Separate this into smaller methods
        old_instance = self.get_object()
        old_user = old_instance.user

        user = self.request.data.get("user_email")
        queue = self.request.data.get("queue_uuid")
        serializer.save()

        if not (user or queue):
            return None

        instance = serializer.instance

        # Create transfer object based on whether it's a user or a queue transfer and add it to the history
        if user:
            if old_instance.user is None:
                time = timezone.now() - old_instance.modified_on
                room_metric = RoomMetrics.objects.get_or_create(room=instance)[0]
                room_metric.waiting_time += time.total_seconds()
                room_metric.queued_count += 1
                room_metric.save()
            else:
                # Get the room metric from instance and update the transfer_count value.
                room_metric = RoomMetrics.objects.get_or_create(room=instance)[0]
                room_metric.transfer_count += 1
                room_metric.save()

            action = "transfer" if self.request.user.email != user else "pick"
            feedback = create_transfer_json(
                action=action,
                from_=old_instance.user or old_instance.queue,
                to=instance.user,
            )

        if queue:
            # Create constraint to make queue not none
            feedback = create_transfer_json(
                action="transfer",
                from_=old_instance.user or old_instance.queue,
                to=instance.queue,
            )
            if (
                not user
            ):  # if it is only a queue transfer from a user, need to reset the user field
                instance.user = None

            room_metric = RoomMetrics.objects.get_or_create(room=instance)[0]
            room_metric.transfer_count += 1
            room_metric.save()

        instance.transfer_history = feedback
        instance.save()

        # Create a message with the transfer data and Send to the room group
        # TODO separate create message in a function
        create_room_feedback_message(instance, feedback, method="rt")

        if old_user is None and user:  # queued > agent
            instance.notify_queue("update")
        elif old_user is not None:
            instance.notify_user("update", user=old_user)
            if queue:  # agent > queue
                instance.notify_queue("update")
            else:  # agent > agent
                instance.notify_user("update")

    def perform_destroy(self, instance):
        instance.notify_room("destroy", callback=True)
        super().perform_destroy(instance)

    @action(
        detail=True,
        methods=["POST", "GET"],
        url_name="chat_completion",
    )
    def chat_completion(self, request, *args, **kwargs) -> Response:
        user_token = request.META.get("HTTP_AUTHORIZATION")
        room = self.get_object()
        if request.method == "GET":
            return Response(
                status=status.HTTP_200_OK,
                data={
                    "can_use_chat_completion": room.queue.sector.can_use_chat_completion
                },
            )
        if not room.queue.sector.can_use_chat_completion:
            return Response(
                status=status.HTTP_400_BAD_REQUEST,
                data={"detail": "Chat completion is not configured for this sector."},
            )
        token = room.queue.sector.project.get_openai_token(user_token)
        if not token:
            return Response(
                status=status.HTTP_400_BAD_REQUEST,
                data={"detail": "OpenAI token not found"},
            )
        messages = room.last_5_messages
        if not messages:
            return Response(
                status=status.HTTP_400_BAD_REQUEST,
                data={"detail": "The selected room does not have messages"},
            )
        serialized_data = ChatCompletionSerializer(messages, many=True).data

        sector = room.queue.sector
        if sector.completion_context:
            serialized_data.append(
                {"role": "system", "content": sector.completion_context}
            )

        openai_client = OpenAIClient()
        completion_response = openai_client.chat_completion(
            token=token, messages=serialized_data
        )
        return Response(
            status=completion_response.status_code, data=completion_response.json()
        )

    @action(
        detail=True,
        methods=["PATCH"],
    )
    def update_custom_fields(self, request, pk=None):
        custom_fields_update = request.data
        data = {"fields": custom_fields_update}

        if pk is None:
            return Response(
                {"Detail": "No room on the request"}, status.HTTP_400_BAD_REQUEST
            )
        elif not custom_fields_update:
            return Response(
                {"Detail": "No custom fields on the request"},
                status.HTTP_400_BAD_REQUEST,
            )

        room = get_editable_custom_fields_room({"uuid": pk, "is_active": "True"})

        custom_field_name = list(data["fields"])[0]
        old_custom_field_value = room.custom_fields.get(custom_field_name, None)
        new_custom_field_value = data["fields"][custom_field_name]

        update_flows_custom_fields(
            project=room.queue.sector.project,
            data=data,
            contact_id=room.contact.external_id,
        )

        update_custom_fields(room, custom_fields_update)

        feedback = {
            "user": request.user.first_name,
            "custom_field_name": custom_field_name,
            "old": old_custom_field_value,
            "new": new_custom_field_value,
        }
        create_room_feedback_message(room, feedback, method="ecf")

        return Response(
            {"Detail": "Custom Field edited with success"},
            status.HTTP_200_OK,
        )

from random import sample
from django.http import HttpResponseServerError
from rest_framework.decorators import action
from rest_framework.viewsets import ViewSet
from rest_framework.response import Response
from rest_framework import serializers, status
from drf_yasg import openapi
from drf_yasg.utils import swagger_auto_schema
from random import choice
import logging

from tarottimeapi.http.openai import OpenaiHandler
from tarottimeapi.serializers import CardSerializer, DailyReadingSerializer, ReadingCardSerializer
from tarottimeapi.models import Reading, Card, ReadingCard, User

logger = logging.getLogger(__name__)

class ReadingView(ViewSet):
    """Tarot Time Readings"""
    @action(detail=False, methods=['post'])
    @swagger_auto_schema(
        manual_parameters=[
            openapi.Parameter('authorization', openapi.IN_HEADER, description="User's UUID", type=openapi.TYPE_STRING)
        ],
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            required=['question'],
            properties={
                'question': openapi.Schema(type=openapi.TYPE_STRING, description='User question')
            }
        )
    )
    def daily_reading(self, request):
        """Handle POST requests for daily reading
        Returns:
            Response -- JSON serialized daily reading instance
            """
        try:
            uid = request.headers.get('authorization')
            user = User.objects.get(uid=uid)
            question = request.data.get('question')
            all_cards = Card.objects.all()
            drawn_cards = sample(list(all_cards), 3)


            reading = Reading.objects.create(
                ques=question,
                user_id=user,
                answer="waiting for openai",
            )
            reading_cards = []
            for order, card in enumerate(drawn_cards):
                is_reversed = choice([True, False])
                reading_card = ReadingCard(reading_id=reading, card_id=card, order=order, is_reversed=is_reversed)
                reading_cards.append(reading_card)
            ReadingCard.objects.bulk_create(reading_cards)
            sorted_reading_cards = ReadingCard.objects.filter(reading_id=reading.id).order_by('order')

            openai_response = OpenaiHandler.call_openai(sorted_reading_cards, question)
            if openai_response == "failure":
                return Response({"message": "OpenAI call failed"}, status=status.HTTP_400_BAD_REQUEST)
            else:
                reading.answer = openai_response
                reading.save()
                serialized_reading_cards = ReadingCardSerializer(sorted_reading_cards, many=True).data


                reading_data = DailyReadingSerializer(reading).data
                reading_data['cards'] = serialized_reading_cards

                return Response(reading_data, status=status.HTTP_201_CREATED)
        except Exception as ex:
            logger.exception("An error occurred while processing the daily reading: %s, user: %s", ex.args[0], uid)
            return HttpResponseServerError(ex)

    @action(detail=False, methods=['get'])
    @swagger_auto_schema(
    operation_description="Description of list method"
    )
    def reading_history_list(self, request):
        """Handle GET requests to get user's previous readings

        Returns:
            Response -- JSON serialized list of readings
        """
        try:
            uid = request.headers.get('authorization')
            user = User.objects.get(uid=uid)
            readings = Reading.objects.filter(user_id=user.id)
            serializer = DailyReadingSerializer(readings, many=True)
            return Response(serializer.data)

        except Exception as ex:
            logger.exception("An error occurred while retrieving reading history: %s, user: %s", ex.args[0], uid)
            return HttpResponseServerError(ex)

    def retrieve(self, request, pk=None):
        try:
            reading = Reading.objects.get(pk=pk)
            serializer = DailyReadingSerializer(reading, many=False)
            return Response(serializer.data)
        except Reading.DoesNotExist:
            # Handle the case where the reading does not exist
            return Response({"message": "Reading not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as ex:
            logger.exception("An error occurred while retrieving the reading: %s", ex.args[0])
            return HttpResponseServerError(ex)

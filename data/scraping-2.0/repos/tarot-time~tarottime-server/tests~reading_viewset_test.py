from unittest.mock import patch

from rest_framework import status
from rest_framework.test import APITestCase
import tarottimeapi
from tarottimeapi.models import  Reading, Card, ReadingCard
from tarottimeapi.serializers import DailyReadingSerializer, ReadingCardSerializer
from tarottimeapi.http.openai import OpenaiHandler

class ReadingViewTestCase(APITestCase):
    def setUp(self):
        self.daily_reading_cards = Card.objects.filter(id__in=[1, 2, 3])


    def test_daily_reading(self):
        with patch.object(OpenaiHandler, 'call_openai', return_value="Your mocked OpenAI response"):
            url = "/reading/daily_reading"
            data = {"question": "Should I buy a new laptop?"}
            # Make a POST request to the daily_reading endpoint
            response = self.client.post(url, data, format="json")

            self.assertEqual(response.status_code, status.HTTP_201_CREATED)

            returned_reading = Reading.objects.get(id=response.data['id'])
            returned_cards = ReadingCard.objects.filter(reading_id=returned_reading.id).order_by('order').all()
            self.assertEqual(len(returned_cards), 3)
            self.assertEqual(returned_reading.ques, data['question'])
            self.assertEqual(returned_reading.answer, "Your mocked OpenAI response")
            serialized_cards = ReadingCardSerializer(returned_cards, many=True).data
            serializer = DailyReadingSerializer(returned_reading).data
            serializer['cards'] = serialized_cards
            self.assertEqual(response.data, serializer)


    def test_reading_history(self):
        url = "/reading/reading_history_list"
        response = self.client.get(url)
        readings_list = Reading.objects.filter(id__in=[1, 2])
        expected = DailyReadingSerializer(readings_list, many=True)
        self.assertEqual(status.HTTP_200_OK, response.status_code)
        self.assertEqual(expected.data, response.data)

    def test_reading_history_retrieve(self):
        url = "/reading/1/reading_history_retrieve"
        response = self.client.get(url)
        reading = Reading.objects.get(id=1)
        expected = DailyReadingSerializer(reading)
        self.assertEqual(status.HTTP_200_OK, response.status_code)
        self.assertEqual(expected.data, response.data)

from django.test import TestCase

# Create your tests here.
import unittest
from unittest import mock
from unittest.mock import MagicMock
import openai
from datetime import datetime
from rest_framework.exceptions import ValidationError
from django.test import RequestFactory
from django.core.serializers.json import DjangoJSONEncoder
from rest_framework.test import force_authenticate
from .algorithm import calculate_distance, calculate_angle, magic
from .models import User, UserPref, UserRoute, Nodes
from rest_framework.test import APIClient
from .serializers import UserSerializer, UserPreferencesSerializer, UserRouteSerializer
from .views import registration, preferences
import json

# TestCases for algorithm.py


def test_magic_park():
    with open('allnode_preferences.json', 'w') as f:
        f.write(
            '{"data_from_frontend": {"selectedOptions": ["park_locations"]}}')

    parks_data = {
        "data": [
            {
                "id": "P410310399",
                "name": "Coleman Playground",
                "type": "park",
                "location": {
                    "latitude": 40.7111652,
                    "longitude": -73.9933311
                },
                "b-score": {
                    "0": -0.4643,
                    "1": -0.5421,
                    "2": -0.6081,
                    "3": -0.6413,
                    "4": -0.6166,
                    "5": -0.5907,
                    "6": -0.429,
                    "7": -0.2606,
                    "8": 0.0896,
                    "9": 0.1137,
                    "10": 0.1339,
                    "11": 0.3008,
                    "12": 0.4307,
                    "13": 0.4469,
                    "14": 0.4839,
                    "15": 0.5511,
                    "16": 0.5717,
                    "17": 0.6442,
                    "18": 0.5086,
                    "19": 0.1926,
                    "20": -0.0171,
                    "21": -0.1165,
                    "22": -0.3392,
                    "23": -0.4483
                }
            },
            # more park dictionaries go here
        ]
    }

    # Initialize the object you want to test
    processor = Processor()

    # Call the method you want to test
    results = processor.magic(parks_data)

    # Assert that the method returns what you expect
    assert isinstance(results, list)
    assert len(results) > 0

    # Cleanup
    os.remove('allnode_preferences.json')


class TestAlgorithm(unittest.TestCase):

    def test_calculate_distance_same_coordinates(self):
        lat1, lon1 = 0, 0
        lat2, lon2 = 0, 0
        result = calculate_distance(lat1, lon1, lat2, lon2)
        self.assertEqual(result, 0)

    def test_calculate_distance(self):
        lat1, lon1 = 52.5200, 13.4050  # Coordinates for Berlin
        lat2, lon2 = 48.8566, 2.3522  # Coordinates for Paris
        result = calculate_distance(lat1, lon1, lat2, lon2)
        # The distance from Berlin to Paris is roughly 878 km.
        self.assertAlmostEqual(result, 878, delta=50)

    def test_calculate_angle_same_coordinates(self):
        lat1, lon1 = 0, 0
        lat2, lon2 = 0, 1  # 1 degree east
        result = calculate_angle(lat1, lon1, lat2, lon2)
        self.assertEqual(result, 90)

    def test_calculate_angle(self):
        lat1, lon1 = 52.5200, 13.4050  # Coordinates for Berlin
        lat2, lon2 = 48.8566, 2.3522  # Coordinates for Paris
        result = calculate_angle(lat1, lon1, lat2, lon2)
        # The bearing from Berlin to Paris is roughly 246 degrees.
        self.assertAlmostEqual(result, 246, delta=10)

# tests for models.py


class UserModelTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        # Set up non-modified objects used by all test methods
        User.objects.create(first_name='Big', last_name='Bob', username='bigbob@gmail.com',
                            address='123 Main St', password='password123')

    def test_first_name_label(self):
        user = User.objects.get(username='bigbob@gmail.com')
        field_label = user._meta.get_field('first_name').verbose_name
        self.assertEqual(field_label, 'First Name')

    def test_last_name_label(self):
        user = User.objects.get(username='bigbob@gmail.com')
        field_label = user._meta.get_field('last_name').verbose_name
        self.assertEqual(field_label, 'Last Name')

    def test_username_label(self):
        user = User.objects.get(username='bigbob@gmail.com')
        field_label = user._meta.get_field('username').verbose_name
        self.assertEqual(field_label, 'Username')

    def test_address_label(self):
        user = User.objects.get(username='bigbob@gmail.com')
        field_label = user._meta.get_field('address').verbose_name
        self.assertEqual(field_label, 'Address')

    def test_password_label(self):
        user = User.objects.get(username='bigbob@gmail.com')
        field_label = user._meta.get_field('password').verbose_name
        self.assertEqual(field_label, 'Password')

    def test_registration_date_label(self):
        user = User.objects.get(username='bigbob@gmail.com')
        field_label = user._meta.get_field('registrationDate').verbose_name
        self.assertEqual(field_label, 'Registration Date')

    def test_first_name_max_length(self):
        user = User.objects.get(username='bigbob@gmail.com')
        max_length = user._meta.get_field('first_name').max_length
        self.assertEqual(max_length, 30)


# # tests for serializers.py

class TestUserSerializer(TestCase):

    def setUp(self):
        self.user_attributes = {
            'first_name': 'Big',
            'last_name': 'Bob',
            'username': 'bigbob@gmail.com',
            'address': '123 Main St',
            'password': 'password123',
            'registrationDate': '2023-08-06'
        }

        self.serializer_data = UserSerializer().data
        self.user = User.objects.create(**self.user_attributes)
        self.serializer = UserSerializer(instance=self.user)

    def test_contains_expected_fields(self):
        data = self.serializer.data
        self.assertCountEqual(data.keys(), [
                              'first_name', 'last_name', 'username', 'address', 'password', 'registrationDate'])

    def test_first_name_field_content(self):
        data = self.serializer.data
        self.assertEqual(data['first_name'],
                         self.user_attributes['first_name'])


class TestUserPreferencesSerializer(TestCase):

    def setUp(self):
        self.user_pref_attributes = {
            'library': True,
            'worship': False,
            'community': True,
            'museum': False,
            'walking_node': True,
            'park_node': False
        }

        self.user_pref = UserPref.objects.create(**self.user_pref_attributes)
        self.serializer = UserPreferencesSerializer(instance=self.user_pref)

    def test_contains_expected_fields(self):
        data = self.serializer.data
        self.assertCountEqual(data.keys(), [
                              'library', 'worship', 'community', 'museum', 'walking_node', 'park_node'])

    def test_library_field_content(self):
        data = self.serializer.data
        self.assertEqual(data['library'], str(
            self.user_pref_attributes['library']))

# # Unit test for sum.py


class TestSum(unittest.TestCase):

    def test_sum(self):
        self.assertEqual(1 + 2, 3)

# Unit test for views.py


class UserViewsTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        # Set up non-modified objects used by all test methods
        User.objects.create(first_name='Big', last_name='Bob', username='bigbob@gmail.com',
                            address='123 Main St', password='password123')

    def setUp(self):
        # Set up client for all the test methods
        self.client = APIClient()

    def test_registration_get(self):
        response = self.client.get('/users/registration')
        users = User.objects.all()
        serializer = UserSerializer(users, many=True)
        self.assertEqual(response.data, serializer.data)
        self.assertEqual(response.status_code, 200)

    def test_registration_post(self):
        data = {'first_name': 'Alice', 'last_name': 'Adams', 'username': 'aliceadams@gmail.com',
                'address': '456 Main St', 'password': 'password456'}
        response = self.client.post('/users/registration', data)
        self.assertEqual(response.status_code, 201)


if __name__ == "__main__":
    unittest.main()

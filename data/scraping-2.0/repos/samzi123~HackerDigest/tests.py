import json
import logging
import os
import unittest
from typing import List
from urllib.parse import urlparse

from fastapi.testclient import TestClient
from requests_mock import Mocker
from readability import Document

from main import app, get_hackernews_top_stories, get_url_text, summarize


class TestApp(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.maxDiff = None

    def test_get_hackernews_top_stories(self):
        # Set up mock response for GET request to topstories.json
        expected_stories = [
            {
                "title": "Example Story 1",
                "url": "https://example.com/story1",
                "score": 100,
            },
            {
                "title": "Example Story 2",
                "url": "https://example.com/story2",
                "score": 200,
            },
        ]
        with Mocker() as m:
            m.get("https://hacker-news.firebaseio.com/v0/topstories.json", json=[1, 2])
            m.get(
                "https://hacker-news.firebaseio.com/v0/item/1.json",
                json={
                    "title": "Example Story 1",
                    "url": "https://example.com/story1",
                    "type": "story",
                    "score": 100,
                },
            )
            m.get(
                "https://hacker-news.firebaseio.com/v0/item/2.json",
                json={
                    "title": "Example Story 2",
                    "url": "https://example.com/story2",
                    "type": "story",
                    "score": 200,
                },
            )
            self.assertEqual(get_hackernews_top_stories(2), expected_stories)

    def test_get_url_text(self):
        # Set up mock response for GET request to example.com/story
        with Mocker() as m:
            m.get(
                "https://example.com/story",
                text="<html><body>Example Story</body></html>",
            )
            self.assertEqual(
                get_url_text("https://example.com/story"), "Example Story "
            )

        # Test response timeout
        with Mocker() as m:
            m.get("https://example.com/story", exc=requests.exceptions.Timeout)
            self.assertIsNone(get_url_text("https://example.com/story"))

        # Test unsuccessful response
        with Mocker() as m:
            m.get("https://example.com/story", status_code=404)
            self.assertIsNone(get_url_text("https://example.com/story"))

        # Test unsupported content type
        with Mocker() as m:
            m.get(
                "https://example.com/story",
                headers={"Content-Type": "image/jpeg"},
                text="<html><body>Example Story</body></html>",
            )
            self.assertIsNone(get_url_text("https://example.com/story"))

        # Test no text content found
        with Mocker() as m:
            m.get(
                "https://example.com/story",
                text='<html><body><script>alert("hello world");</script></body></html>',
            )
            self.assertIsNone(get_url_text("https://example.com/story"))

    def test_summarize(self):
        # Set up mock response from OpenAI API
        with Mocker() as m:
            m.post(
                "https://api.openai.com/v1/chat-completions",
                json={
                    "choices": [{"message": {"content": "Summary of Example Text."}}]
                },
            )
            self.assertEqual(summarize("Example Text."), "Summary of Example Text.")

        # Test unsuccessful response from OpenAI API
        with Mocker() as m:
            m.post("https://api.openai.com/v1/chat-completions", status_code=500)
            self.assertEqual(summarize("Example Text."), "")

    def test_main():
        with TestClient(app) as client:
            # Test the main endpoint
            response = client.get("/")
            assert response.status_code == 200
            assert isinstance(response.json(), list)
            assert len(response.json()) == 10
            for story in response.json():
                assert "title" in story
                assert "url" in story
                assert "score" in story
                assert "text" in story
                assert "summary" in story
                assert isinstance(story["title"], str)
                assert isinstance(story["url"], str)
                assert urlparse(story["url"]).scheme in ["http", "https"]
                assert isinstance(story["score"], int)
                assert isinstance(story["text"], str)
                assert isinstance(story["summary"], str)

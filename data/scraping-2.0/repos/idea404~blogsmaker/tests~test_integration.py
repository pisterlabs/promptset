from unittest import TestCase, skip

from apis import OpenAIClient
from db import BlogSite
from manager import OpenAIManager


class TestIntegrationTests(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.client: OpenAIClient = OpenAIClient()
        self.openai_manager = OpenAIManager(self.client)

    def tearDown(self) -> None:
        super().tearDown()
        self.client = None  # type: ignore

    def test_openai_client(self):
        prompt = "Say this in Dutch: Monday"
        response = self.client.create_completion(prompt, temperature=0)
        self.assertTrue(response)
        self.assertEqual(response, "\n\nMaandag")

    def test_openai_manager_request_article_topics(self):
        site = BlogSite("Cat Care")
        articles = self.openai_manager._request_article_topics(site.subject, 2)
        self.assertTrue(articles)
        self.assertEqual(len(articles), 2)
        for article in articles:
            self.assertTrue(article)
            self.assertEqual(article.topic, article.topic.strip())
            self.assertFalse("\n" in article.topic)

    def test_openai_manager_request_article_text(self):
        subject = "Cat Care"
        topic = "Feeding"
        article_text = self.openai_manager._request_article_text(subject, topic, 1900)
        self.assertTrue(article_text)
        spaced_text = article_text.replace("\n", " ")
        word_count = len(spaced_text.split(" "))
        self.assertTrue(700 <= word_count <= 2200)

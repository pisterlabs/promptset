import json
import logging
from datetime import datetime
from typing import List, NamedTuple, Optional, Tuple

import constants as c
import pinecone
from constants import PINECONE_KEY
from graphqlclient import GraphQLClient
from openai_client import embed_api
from story import Article, Story

OPENAI_EMBEDDING_SIZE = 1536
STORY_NAMESPACE = "story"

main_logger = logging.getLogger(c.MAIN_LOGGER_NAME)


class Topic(NamedTuple):
    id: str
    title: str
    prompt: str


class StoryDatabase:

    CLUSTER_THRESHOLD = 0.9

    def __init__(self):
        self.client = GraphQLClient(
            'http://helmet-ai.azurewebsites.net/graphql')

        pinecone.init(api_key=PINECONE_KEY,
                      environment="asia-southeast1-gcp-free")
        self.index = pinecone.Index("helmet")

    def find_most_similar_story(self,
                                embedding: List[float]) -> Optional[Story]:
        matches = self.index.query(top_k=1,
                                   vector=embedding,
                                   namespace=STORY_NAMESPACE).matches
        if len(matches) == 0:
            return None
        query_response = matches[0]
        if query_response.score > self.CLUSTER_THRESHOLD:
            query = '''
                query Query($id: ID!) {
                    story(id: $id) {
                        title
                        body
                        citationStories {
                            id
                            citation {
                                title
                                body
                                url
                            }
                        }
                    }
                }
            '''
            # Set the variables for the query
            variables = {'id': query_response.id}

            # Execute the query
            result = self.client.execute(query, variables)

            # Extract the story and citation data from the response
            response_data = json.loads(result)
            story_data = response_data['data']['story']
            title = story_data['title']
            body = story_data['body']
            citations = [
                data['citation'] for data in story_data['citationStories']
            ]

            return Story.from_tuple(title, body, citations)

    def _upsert_story(self, story: Story) -> None:
        embed = embed_api(f"{story.title} {story.summary}")
        self.index.upsert([(story.id, embed, {
            "timestamp": datetime.utcnow().isoformat()
        })],
                          namespace=STORY_NAMESPACE)

    def _replace_story(self, story: Story) -> str:
        mutation = """
        mutation($input: StoryInput!) {
            createStory(input: $input) {
                id
            }
        }
        """
        variables = {"input": {'title': story.title, 'body': story.summary}}
        result = self.client.execute(mutation, variables)
        response_data = json.loads(result)
        story.id = response_data['data']['createStory']['id']

    def add_story(self, story: Story) -> None:
        self._replace_story(story)
        self._upsert_story(story)

    def update_story(self, story: Story, article: Article) -> None:
        story.add_source(article)
        self._replace_story(story)
        self._upsert_story(story)

    def process_article(self, article: Article) -> Tuple[str, List[float]]:
        mutation = """
        mutation($input: CitationInput!) {
            createCitation(input: $input) {
                id
                title
                body
                url
            }
        }
        """
        variables = {
            'input': {
                'title': article.title,
                'body': article.text,
                'url': article.link
            }
        }
        result = self.client.execute(mutation, variables)
        response_data = json.loads(result)
        article_id = response_data['data']['createCitation']['id']
        return article_id, embed_api(article.text)

    def add_article_to_story(self, story_id: str, article_id: str) -> None:
        mutation = """
        mutation($storyId: ID!, $citationId: ID!) {
            createCitationStory(storyId: $storyId, citationId: $citationId, explanation: "") {
                id
            }
        }
        """
        variables = {'storyId': story_id, 'citationId': article_id}
        self.client.execute(mutation, variables)

    def add_story_to_topic(self, story_id: str, topic_id: str,
                           explanation: str) -> None:
        mutation = """
        mutation($storyId: ID!, $topicId: ID!, $explanation: String!) {
            createStoryTopic(storyId: $storyId, topicId: $topicId, explanation: $explanation) {
                id
            }
        }
        """
        variables = {
            "storyId": story_id,
            "topicId": topic_id,
            "explanation": explanation
        }
        self.client.execute(mutation, variables)

    def get_all_topics(self) -> List[Topic]:
        query = """
        query GetTopics {
            topics {
                id
                title
                prompt
            }
        }
        """
        result = self.client.execute(query)
        response_data = json.loads(result)
        if 'errors' in response_data:
            # Handle errors
            main_logger.error("GraphQL query failed: %s",
                              response_data['errors'])
            return []
        return [
            Topic(topic['id'], topic['title'], topic['prompt'])
            for topic in response_data['data']['topics']
        ]

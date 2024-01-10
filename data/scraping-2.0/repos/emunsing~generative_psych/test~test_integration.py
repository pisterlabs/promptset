from typing import Any
from collections import OrderedDict
from generative_psych import ConversationAgent, RelationshipConversationContext
from datastore import DummyNarrativeStore, FileNarrativeStore
from psych_helpers import OpenAIQueryAPI
import tempfile
import logging
logging.basicConfig(level=logging.DEBUG)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel("DEBUG")

ALICE = dict(name='Alice',
             background="Alice is a startup founder in her mid-twenties",
             relationship_goals="Find someone who challenges them",
             other_major_goals="Grow their business, have fun",
             personal_context="just got off work and is looking forward to the evening",
            )

BOB = dict(name='Bob',
           background="Bob is a nervous recent college graduate",
           relationship_goals="Find a partner",
           other_major_goals="Express themselves creatively, find friendships, develop roots",
           personal_context="just got off work and is looking forward to the evening",
           )

class conditional_reflection_api:
    def __call__(self, *args: Any, **kwds: Any) -> Any:
            possible_responses = [('You are writing a dialogue between', ["Bob: Hi Alice!", 'Alice: Hi Bob, how are you?']),
                                  ('Output a line beginning with "Relationship goals:"', ["Relationship goals: get married"]),
                                  ('Output a line beginning with "Other goals:"', ["Other goals: care for their parents"]),
                                  ('First, print "Feelings:" with a few words', ['Feelings: angry and resentful for being lied to']),
                                  ('First, output "Emotions:"', ['Emotions: angry, resentful']),
                                  ('Then on a new line output "Needs:"', ['Needs: trust, support']),
                                  ('both want to break up', ['END NARRATIVE']),
                                  ]
            response_lines = []
            query = args[0]
            LOGGER.debug(f"Query: {args[0]}")

            for query_key, query_response in possible_responses:
                 if query_key in query:
                        response_lines += query_response

            assert len(response_lines) > 0, f"Query did not match any possible responses"
            response = "\n".join(response_lines)

            LOGGER.info(f"Response: {response}")
            return response

class RelationshipConversationContextTest(RelationshipConversationContext):
    def __init__(self, query_api, narrative_store, person1, person2) -> None:
        super().__init__(query_api, narrative_store, person1, person2)
        self.reflection_interval = 2
        self.max_rounds_in_conversation = 3

    @property
    def chapter_progression_rate(self):
        return OrderedDict([("are on their first date", 1.0),  # Progression = probability of changing state
                            ("recently started dating", 1.0),
                            ("both want to break up", 0.0),
                            ])

def test_integration():
    query_api = conditional_reflection_api()
    p1 = ConversationAgent(query_api=query_api, **ALICE)
    p2 = ConversationAgent(query_api=query_api, **BOB)
    
    with tempfile.TemporaryDirectory() as tempdir:
        store = FileNarrativeStore(data_dir=tempdir)
        runner = RelationshipConversationContextTest(query_api, store, p1, p2)

        runner.run_relationship()
        print("Done running test conversation")

def test_integration_openai():
    query_api = OpenAIQueryAPI(temperature=0.0)
    p1 = ConversationAgent(query_api=query_api, **ALICE)
    p2 = ConversationAgent(query_api=query_api, **BOB)
    
    with tempfile.TemporaryDirectory() as tempdir:
        store = FileNarrativeStore(data_dir=tempdir)
        runner = RelationshipConversationContextTest(query_api, store, p1, p2)
        runner.run_relationship()
        print("Done running test conversation")

from typing import Dict, Text, Any, List

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData

import os
import json
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

prompt_template = """
You are part of the llm-based entity extraction component used in a Rasa-based dialog engine.

Please use the format that Rasa uses for entities 
{{"entity": "entity_name", "value": "entity_value", "start": 0, "end": 4, "extractor": "llm_entity_extractor_component"}}.
Return the entities as a JSON array.

You should extract the following entities wrapped in brackets [].
{0}

The items in the entity list can also include a description of the entity in the following format.
entity_name "entity_description"
You can use the description to help you extract the entity.

Please extract the entities from the following text provided as user message and return them as a JSON array.

"""

@DefaultV1Recipe.register(
    [DefaultV1Recipe.ComponentType.ENTITY_EXTRACTOR], is_trainable=False
)
class EntityExtractor(GraphComponent):
    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:

        return cls(config)
    
    def __init__(self, config: Dict[Text, Any]):

        super().__init__()

        if "entities" in config:
            self.entities = config["entities"]
        else:
            self.entities = []

        if "model" in config:
            self.model = config["model"]
        else:
            self.model = "gpt-3.5-turbo"

        if "temperature" in config:
            self.temperature = config["temperature"]
        else:
            self.temperature = 0
        
        if "max_tokens" in config:
            self.max_tokens = config["max_tokens"]
        else:
            self.max_tokens = 250

        self.prompt = prompt_template.format(self.entities)


    def train(self, training_data: TrainingData) -> Resource:
        pass

    def process_training_data(self, training_data: TrainingData) -> TrainingData:

        return training_data

    def process(self, messages: List[Message]) -> List[Message]:
        # This is the method which Rasa Open Source will call during inference.
        
        for message in messages:
            message = self.extract_entities_with_llm(message)

        return messages
    
    def extract_entities_with_llm(self, message: Message) -> Message:

        text = message.get("text")

        # send text to LLM model

        # The main disadvanteges of the LLM models are that the response time is slow.
        chat_completion = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": text}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        # get entities from LLM model

        entities = chat_completion.choices[0]['message']['content']

        entities_dict = json.loads(entities)

        message.set("entities", entities_dict)

        return message
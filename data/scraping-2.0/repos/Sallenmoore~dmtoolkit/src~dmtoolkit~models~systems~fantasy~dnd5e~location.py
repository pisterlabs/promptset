import json
import random

from autonomous import log
from autonomous.apis import OpenAI
from autonomous.model.automodel import AutoModel
from autonomous.storage.cloudinarystorage import CloudinaryStorage
from dmtoolkit.models.base.location import Location
from .character import DnDCharacter


class DnDLocation(AutoModel):
    def add_inhabitant(self, character=None):
        if not character:
            character = DnDCharacter.generate()
            character.save()
        if character not in self.inhabitants:
            self.inhabitants.append(character)
            self.save()

    def get_image_prompt(self):
        description = (
            self.desc
            or "An illustraion of a terrifying or wonderful building or landmark in a D&D5e world."
        )

        return f"A full color illustrated interior image of a building or landmark in a D&D5e campaign called {self.name} with the following description: {description}"

    @classmethod
    def generate(cls, primer=None, prompt=None):
        primer = (
            primer
            or """
        You are an expert worldbuilding AI that creates interesting and varied locations to explore for a D&D5e campaign.
        """
        )
        prompt = (
            prompt
            or "Generate a random structure or landmark of interest for a D&D5e campaign."
        )
        prompt += "with the following attributes:\n\nName: \nType: \nDescription: \nInventory: \n"

        return residence_obj


class Shop(Location):
    def get_image_prompt(self):
        description = (
            self.desc or "A simple general goods shop with wooden counters and shelves."
        )

        return f"A full color interior image of a medieval fantasy merchant shop called {self.name} with the following description: {description}"

    @classmethod
    def generate(cls, primer=None, prompt=None):
        primer = (
            primer
            or """
        You are a fictional shop generator that creates appropriate business establishments for a fictional world.
        """
        )
        prompt = prompt or "Generate a random business establishment"
        prompt += "with the following attributes:\n\nName: \nType: \nDescription: \nInventory: \n"
        funcobj = {
            "name": "generate_shop",
            "description": "builds an Shop model object",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The shop's name",
                    },
                    "type": {
                        "type": "string",
                        "description": "The type of wares the shop sells",
                    },
                    "desc": {
                        "type": "string",
                        "description": "A short description of the inside of the shop",
                    },
                    "inventory": {
                        "type": "array",
                        "description": "The shop's inventory of purchasable items",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "The name of the item",
                                },
                                "desc": {
                                    "type": "string",
                                    "description": "A short description of the item",
                                },
                                "cost": {
                                    "type": "string",
                                    "description": "the cost of the item",
                                },
                            },
                        },
                    },
                },
            },
        }

        funcobj["parameters"]["required"] = list(
            funcobj["parameters"]["properties"].keys()
        )
        shop = OpenAI().generate_text(prompt, primer, functions=funcobj)
        try:
            shop = json.loads(shop)
        except Exception as e:
            log(e)
            return None
        else:
            shopowner = Character.generate(
                summary=f"Owner of {shop['name']}, a {shop['shoptype']} business."
            )
            shopowner.save()
            shop["owner"] = shopowner
            shop_obj = Shop(**shop)
            shop_obj.save()
        return shop_obj


class Residence(Location):
    def get_image_prompt(self):
        description = self.desc or "A simple residence for a fictional character."

        return f"A full color illustrated interior image of a residence {self.name} with the following description: {description}"

    @classmethod
    def generate(cls, primer=None, prompt=None):
        primer = (
            primer
            or """
        You are an expert worldbuilding AI that creates interesting residential buildings for a fictional world.
        """
        )
        prompt = prompt or "Generate a random residential building of interest"
        prompt += "with the following attributes:\n\nName: \nType: \nDescription: \nInventory: \n"
        funcobj = {
            "name": "generate_shop",
            "description": "builds an Shop model object",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The residences's name",
                    },
                    "type": {
                        "type": "string",
                        "description": "The type of residence",
                    },
                    "desc": {
                        "type": "string",
                        "description": "A short description of the inside of the residence",
                    },
                    "inventory": {
                        "type": "array",
                        "description": "The residence's inventory of valuable items",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "The name of the item",
                                },
                                "desc": {
                                    "type": "string",
                                    "description": "A short description of the item",
                                },
                                "value": {
                                    "type": "string",
                                    "description": "the value of the item",
                                },
                            },
                        },
                    },
                },
            },
        }

        funcobj["parameters"]["required"] = list(
            funcobj["parameters"]["properties"].keys()
        )
        residence = OpenAI().generate_text(prompt, primer, functions=funcobj)
        try:
            residence = json.loads(residence)
        except Exception as e:
            log(e)
            return None
        else:
            owner = Character.generate(
                summary=f"Owner of {residence['name']}, a {residence['shoptype']} business."
            )
            owner.save()
            residence["owner"] = owner
            residence_obj = Residence(**residence)
            residence_obj.inhabitants = [owner]
            residence_obj.save()
        return residence_obj

from __future__ import annotations


import unittest
from dataclasses import dataclass
from typing import Any


from cohere import responses

from clippy.controllers.apis.cohere_controller import CohereController
from clippy.controllers.apis.cohere_controller_utils import Generation, Generations, TokenLikelihood
from clippy.dm.db_utils import Database


class TestModel(unittest.IsolatedAsyncioTestCase):
    async def test_cohere(self):
        db = Database("test_output/test_data.json")
        controller = CohereController()

        embedding = await controller.embed("embed a response")
        generation = await controller.generate(prompt="Generate a response")
        print(generation)
        # print(embedding)

        await controller.close()
        breakpoint()

        self.assertTrue(len(db) == 2)

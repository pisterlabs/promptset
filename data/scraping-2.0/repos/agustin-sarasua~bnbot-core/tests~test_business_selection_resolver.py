import unittest
from unittest.mock import MagicMock
from app.model import Message
from app.task_resolver.engine import StepData
from app.task_resolver.step_resolvers import BusinessSelectionResolver
import json
import openai
from dotenv import load_dotenv, find_dotenv
from unittest.mock import patch
import os

def mock_list_businesses_side_effect(*args, **kwargs):
    # Extracting input arguments
    load_businesses = json.loads(args[0])

    fake_businesses = []
    # Open the file and load its contents as JSON
    with open("data/businesses.json", 'r') as file:
        fake_businesses = json.load(file)

    if load_businesses.get("bnbot_id", None) is not None:
        for fb in fake_businesses:
            if load_businesses.get("bnbot_id", "") == fb["bnbot_id"]:
                return [fb]
        return []
    else:
        return fake_businesses

class TestBusinessSelectionResolver(unittest.TestCase):
    
    def setUp(self):
        _ = load_dotenv(find_dotenv(filename="../.env")) # read local .env file
        openai.api_key = os.environ['OPENAI_API_KEY']
    
    @patch('app.integrations.BackendAPIClient.list_businesses')
    def test_run(self, mock_list_businesses):
        mock_list_businesses.side_effect = mock_list_businesses_side_effect

        # Arrange
        prev_step_data = StepData()
        prev_step_data.resolver_data = {'step_first_execution': False, 'business_info': {'location': 'Mercedes', 'business_name': 'Enrique Joaquín', 'have_enough_info': True}}

        test_cases = [
            {
                "messages": [
                    Message("user",  "Hola"),
                    Message("assistant",  "Hola, ¿en qué puedo ayudarte?"),
                    Message("user",  "Me gustaría reservar una casa"),
                    Message("assistant",  "¡Claro! Estoy aquí para ayudarte a encontrar una casa para reservar. ¿Tienes algún ID de negocio en mente o necesitas ayuda para encontrar uno?"),
                    Message("user",  "No tengo el ID, pero es el complejo Enrique Joaquín"),
                    Message("assistant",  "Perfecto, ¿en qué ubicación te gustaría encontrar el complejo Enrique Joaquín? "),
                    Message("user",  "En Mercedes"),
                ],
                "previous_setp_data": {
                    "GATHER_BUSINESS_INFO": prev_step_data
                },
                "expected_user_has_selected": False,
                "expected_business_id": None,
                "expected_bnbot_id": None
            },
            {
                "messages": [
                    Message("user",  "Hola"),
                    Message("assistant",  "Hola, ¿en qué puedo ayudarte?"),
                    Message("user",  "Me gustaría reservar una casa"),
                    Message("assistant",  "¡Claro! Estoy aquí para ayudarte a encontrar una casa para reservar. ¿Tienes algún ID de negocio en mente o necesitas ayuda para encontrar uno?"),
                    Message("user",  "No tengo el ID, pero es el complejo Enrique Joaquín"),
                    Message("assistant",  "Perfecto, ¿en qué ubicación te gustaría encontrar el complejo Enrique Joaquín? "),
                    Message("user",  "En Mercedes"),
                    Message("assistant", '¡Genial! He encontrado el complejo Enrique Joaquín en Mercedes, Soriano, Uruguay. ¿Es este el negocio que estás buscando?\n\n- Nombre del negocio: Complejo Enrique Joaquín\n- Dirección: Ruta 2 km 284, Mercedes, Soriano, Uruguay\n- Código postal: 75000\n\n¿Es este el negocio que estás buscando?'),
                    Message("user",  "Si"),
                ],
                "previous_setp_data": {
                    "GATHER_BUSINESS_INFO": prev_step_data
                },
                "expected_user_has_selected": True,
                "expected_business_id": "complejo_enrique_joaquin_id",
                "expected_bnbot_id": "@complejo.enrique.joaquin"
            }
        ]
        for idx, test in enumerate(test_cases):
            print(f"Running test {idx}")
            resolver = BusinessSelectionResolver("http://test")
            resolver.data["step_first_execution"] = False
            # Act
            result = resolver.run(test["messages"], test["previous_setp_data"])
            
            # Assert
            self.assertIsNotNone(result)
            if "business_info" in resolver.data:
                self.assertEqual(resolver.data["business_info"]["user_has_selected"], test["expected_user_has_selected"])
                if resolver.data["business_info"]["user_has_selected"]:
                    self.assertEqual(resolver.data["business_info"]["business_id"], test["expected_business_id"])
                    self.assertEqual(resolver.data["business_info"]["bnbot_id"], test["expected_bnbot_id"])
            else: 
                self.assertEqual(resolver.is_done(), False)



if __name__ == '__main__':
    unittest.main()

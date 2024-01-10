import unittest
from unittest.mock import MagicMock
from app.model import Message, StepData
from app.task_resolver.step_resolvers import HouseSelectionResolver

import openai
from dotenv import load_dotenv, find_dotenv
import os

class TestHouseSelectionResolver(unittest.TestCase):
    
    def setUp(self):
        _ = load_dotenv(find_dotenv(filename="../.env")) # read local .env file
        openai.api_key = os.environ['OPENAI_API_KEY']
    
    def test_run(self):
        # Arrange
        prev_step_data = StepData()
        prev_step_data.resolver_data = {'booking_information': {'check_in_date': '2023-06-29', 'check_out_date': '2023-07-01', 'num_guests': 2, 'num_nights': 2}}
        test_cases = [
            # {
            #     "messages": [
            #         Message("user", "Me gustaría reservar una casa para dos personas, para el jueves que viene."),
            #         Message("assistant", "¡Hola! Claro, para poder ayudarte necesito saber la fecha de salida. ¿Cuándo te gustaría dejar la casa? "),
            #         Message("user", "El sábado."),
            #         Message("assistant", "Actualmente tenemos dos propiedades disponibles para las fechas que solicitaste:    1. Cabaña 'Sol': Impresionante villa con vistas panorámicas a las montañas. Esta lujosa propiedad ofrece un ambiente tranquilo y relajante con amplios espacios interiores y exteriores. Cuenta con una piscina privada, jardines exuberantes y una terraza para disfrutar de las maravillosas vistas. Perfecta para escapadas en familia o con amigos. Amenidades: Wi-Fi, estacionamiento privado, se admiten mascotas, barbacoa, piscina privada. Precio por noche: 250.0 USD    2. Cabaña 'Luna': Impresionante villa con vistas panorámicas a las montañas. Esta lujosa propiedad ofrece un ambiente tranquilo y relajante con amplios espacios interiores y exteriores. Cuenta con una piscina privada, jardines exuberantes y una terraza para disfrutar de las maravillosas vistas. Perfecta para escapadas en familia o con amigos. Amenidades: Wi-Fi, estacionamiento privado, se admiten mascotas, barbacoa, piscina privada. Precio por noche: 120.0 USD    ¿Te gustaría reservar alguna de estas propiedades?"),
            #         Message("user", "Se puede hacer asado?"),
            #     ],
            #     "previous_setp_data": {
            #         "GATHER_BOOKING_INFO": prev_step_data
            #     },
            #     "expected_property_id": None
            # },
            # {
            #     "messages": [
            #         Message("user", "Me gustaría reservar una casa para dos personas, para el jueves que viene."),
            #         Message("assistant", "¡Hola! Claro, para poder ayudarte necesito saber la fecha de salida. ¿Cuándo te gustaría dejar la casa? "),
            #         Message("user", "El sábado."),
            #         Message("assistant", "Actualmente tenemos dos propiedades disponibles para las fechas que solicitaste:    1. Cabaña 'Sol': Impresionante villa con vistas panorámicas a las montañas. Esta lujosa propiedad ofrece un ambiente tranquilo y relajante con amplios espacios interiores y exteriores. Cuenta con una piscina privada, jardines exuberantes y una terraza para disfrutar de las maravillosas vistas. Perfecta para escapadas en familia o con amigos. Amenidades: Wi-Fi, estacionamiento privado, se admiten mascotas, barbacoa, piscina privada. Precio por noche: 250.0 USD    2. Cabaña 'Luna': Impresionante villa con vistas panorámicas a las montañas. Esta lujosa propiedad ofrece un ambiente tranquilo y relajante con amplios espacios interiores y exteriores. Cuenta con una piscina privada, jardines exuberantes y una terraza para disfrutar de las maravillosas vistas. Perfecta para escapadas en familia o con amigos. Amenidades: Wi-Fi, estacionamiento privado, se admiten mascotas, barbacoa, piscina privada. Precio por noche: 120.0 USD    ¿Te gustaría reservar alguna de estas propiedades?"),
            #         Message("user", "La primera"),
            #     ],
            #     "previous_setp_data": {
            #         "GATHER_BOOKING_INFO": prev_step_data
            #     },
            #     "expected_property_id": "Sol"
            # },
            {
                "messages": [
                    Message("user", "Hola, necesito una casa para 2 personas por el fin de semana"),
                    Message("assistant", "¡Hola! Tenemos dos opciones disponibles para el fin de semana: la Cabaña \"Sol\" y la Cabaña \"Luna\". Ambas tienen capacidad para más personas, pero también pueden ser reservadas para dos personas. ¿Te gustaría saber más sobre ellas?"),
                    Message("user", "Si")
                ],
                "previous_setp_data": {
                    "GATHER_BOOKING_INFO": prev_step_data
                },
                "expected_property_id": None
            },
            {
                "messages": [
                    Message("user", "Hola, necesito una casa para 2 personas por el fin de semana"),
                    Message("assistant", "¡Hola! Tenemos dos opciones disponibles para el fin de semana: la Cabaña \"Sol\" y la Cabaña \"Luna\". Ambas tienen capacidad para más personas, pero también pueden ser reservadas para dos personas. ¿Te gustaría saber más sobre ellas?"),
                    Message("user", "Quiero reservar la primera")
                ],
                "previous_setp_data": {
                    "GATHER_BOOKING_INFO": prev_step_data
                },
                "expected_property_id": "Sol"
            },
        ]
        
        for idx, test in enumerate(test_cases):
            print(f"Running test {idx}")
            resolver = HouseSelectionResolver()
            resolver.data["step_first_execution"] = False
            # Act
            result = resolver.run(test["messages"], test["previous_setp_data"])
            
            # Assert
            self.assertIsNotNone(result)
            if test["expected_property_id"] is None:
                self.assertTrue("property_picked_info" not in resolver.data or 
                                ("property_picked_info" in resolver.data and "property_id" not in resolver.data["property_picked_info"]))
            else:
                self.assertEqual(resolver.data["property_picked_info"]["property_id"], test["expected_property_id"])



if __name__ == '__main__':
    unittest.main()

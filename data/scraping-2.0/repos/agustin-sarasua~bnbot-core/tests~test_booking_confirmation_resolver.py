import unittest
from unittest.mock import MagicMock
from app.model import Message, StepData
from app.task_resolver.step_resolvers import BookingConfirmationResolver
import os 
import openai
from dotenv import load_dotenv, find_dotenv

class TestBookingConfirmationResolver(unittest.TestCase):
    
    def setUp(self):
        _ = load_dotenv(find_dotenv(filename="../.env")) # read local .env file
        openai.api_key = os.environ['OPENAI_API_KEY']
    
    def test_run(self):
        # Arrange
        def _create_step_data(info) -> StepData:
            step_data = StepData()
            step_data.resolver_data =info
            return step_data
        
        test_cases = [
            {
                "messages": [
                    Message("user", "Me gustaría reservar una casa para dos personas, para el jueves que viene"),
                    Message("assistant", "¡Hola! Claro, para poder ayudarte necesito saber la fecha de salida. ¿Cuándo te gustaría dejar la casa?"),
                    Message("user", "Sería hasta el lunes"),
                    Message("assistant", "Perfecto, tenemos dos opciones disponibles para esas fechas. La primera es la Cabaña \"Sol\" que tiene capacidad para 8 personas y un precio de 250 USD por noche. La segunda opción es la Cabaña \"Luna\" que tiene capacidad para 4 personas y un precio de 120 USD por noche. ¿Te gustaría saber más sobre alguna de estas opciones o prefieres reservar directamente alguna de ellas?"),
                    Message("user", "Quiero reservar las más barata que acepte mascotas"),
                    Message("assistant", "¡Genial! La Cabaña \"Luna\" es pet-friendly y tiene un precio de 120 USD por noche. ¿Te gustaría reservarla?"),
                    Message("user", "Si, perfect"),
                    Message("assistant", "¡Genial! La Cabaña 'Luna' es pet-friendly y tiene un precio de 120 USD por noche. ¿Podrías proporcionarme tu nombre y correo electrónico para proceder con la reserva?"),
                    Message("user", "agustinsarasua@gmail.com, Agustin"),
                ],
                "previous_setp_data": {
                    "GATHER_BOOKING_INFO": _create_step_data({"booking_information":{'check_in_date': '2023-06-29', 'check_out_date': '2023-07-01', 'num_guests': 2, 'num_nights': 2}}),
                    "HOUSE_SELECTION": _create_step_data({'step_first_execution': False, 'properties_available': {'Sol': {'property_id': 'Sol', 'name': 'Cabaña "Sol"', 'description': 'Impresionante villa con vistas panorámicas a las montañas. Esta lujosa propiedad ofrece un ambiente tranquilo y relajante con amplios espacios interiores y exteriores. Cuenta con una piscina privada, jardines exuberantes y una terraza para disfrutar de las maravillosas vistas. Perfecta para escapadas en familia o con amigos.', 'amenities': ['Wi-Fi', 'private parking', 'pet-friendly', 'barbecue', 'private pool'], 'price': '250.0', 'currency': 'USD', 'max_guests': '8', 'how_to_arrive_instructions': 'El complejo queda en ruta 2 km 287, cerca del pejae. Yendo para Fray Bentos desde Mercedes a mano izquierda. Aqui esta la ubicacion en google maps: https://goo.gl/maps/R8gQZDHVXr2tiPQA8', 'pick_up_keys_instructions': 'Las llaves se encuentran en un box en la puerta de entrada. La clave para abrir el box es 12345.'}, 'Luna': {'property_id': 'Luna', 'name': 'Cabaña "Luna"', 'description': 'Impresionante villa con vistas panorámicas a las montañas. Esta lujosa propiedad ofrece un ambiente tranquilo y relajante con amplios espacios interiores y exteriores. Cuenta con una piscina privada, jardines exuberantes y una terraza para disfrutar de las maravillosas vistas. Perfecta para escapadas en familia o con amigos.', 'amenities': ['Wi-Fi', 'private parking', 'pet-friendly', 'barbecue', 'private pool'], 'price': '120', 'currency': 'USD', 'max_guests': '4', 'how_to_arrive_instructions': 'El complejo queda en ruta 2 km 287, cerca del pejae. Yendo para Fray Bentos desde Mercedes a mano izquierda. Aqui esta la ubicacion en google maps: https://goo.gl/maps/R8gQZDHVXr2tiPQA8', 'pick_up_keys_instructions': 'Las llaves se encuentran en un box en la puerta de entrada. La clave para abrir el box es 12345.'}}, 'property_picked_info': {'property_id': 'Luna', 'price_per_night': 'USD 120.0', 'total_price': 'USD 480.0'}}),
                    "GATHER_USER_INFO": _create_step_data({'step_first_execution': True, 'user_information': {'user_name': '', 'email': 'Please provide your name and email to proceed with the booking.', 'text': "¡Genial! La Cabaña 'Luna' es pet-friendly y tiene un precio de 120 USD por noche. ¿Podrías proporcionarme tu nombre y correo electrónico para proceder con la reserva?"}})
                },
                "step_chat_history": [
                    Message("user", "agustinsarasua@gmail.com, Agustin")
                ],
                "expected_resolver_done": False
            }
        ]
        
        for idx, test in enumerate(test_cases):
            print(f"Running test {idx}")
            resolver = BookingConfirmationResolver()

            # Act
            result = resolver.run(test["messages"], test["previous_setp_data"], test["step_chat_history"])
            
            # Assert
            self.assertIsNotNone(result)
            self.assertEqual(test["expected_resolver_done"], resolver.is_done())



if __name__ == '__main__':
    unittest.main()

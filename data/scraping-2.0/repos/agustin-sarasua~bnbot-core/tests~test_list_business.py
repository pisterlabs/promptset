import unittest
from unittest.mock import MagicMock
from app.backend.domain.entities.business import BusinessOwner, LoadBusinesses, Business, Location, PaymentOption, Property
from app.model import Message
from app.backend.domain.usecases.list_business import ListBusinessUseCase
from app.backend.infraestructure.repositories import BusinessRepository
import openai
from dotenv import load_dotenv, find_dotenv
from unittest.mock import MagicMock
import json
import os

class TestListBusinessUseCase(unittest.TestCase):
    
    def setUp(self):
        _ = load_dotenv(find_dotenv(filename="../.env")) # read local .env file
        openai.api_key = os.environ['OPENAI_API_KEY']
    
    def test_execute(self):
        # Arrange
        test_cases = [
            # {
            #     "load_business": LoadBusinesses(bnbot_id="@complejo.enrique.joaquin", location="Mercedes"),
            #     "expected_resolver_done": False
            # },
            {
                "load_business": LoadBusinesses(bnbot_id='', location='Mercedes', business_name='complejo Joaquin', business_owner=''),
                "expected_resolver_done": False
            }
        ]
        # fake_businesses = []
        # # Open the file and load its contents as JSON
        # with open("data/businesses.json", 'r') as file:
        #     fake_businesses = json.load(file)
        
        # businesses = []
        # for fb in fake_businesses:
        #     business = Business(**fb)
        #     businesses.append(business)
        businesses = [
            Business(business_id='f755f3d3-9261-46be-9a7c-fdb7d93779ec', 
                    business_name='Rio Claro Campgrounds', 
                    description='An eco-friendly campsite along the Rio Claro river, ideal for nature lovers.', 
                    bnbot_id='@rio_claro_campgrounds', 
                    bnbot_configuration={'booking_deposit': 0.1, 'pre_book_time': 48.0}, 
                    location=Location(latitude=37.7749, longitude=-122.4194, address='Ruta 2 km 284', city='Mercedes', state='Soriano', country='Uruguay', postal_code='75000'), 
                    business_owners=[BusinessOwner(name='Miguel Alvarez', phone_number='+59891231234', email='miguel.alvarez@example.com')], 
                    payment_options=[PaymentOption(payment_method='MOBILE_PAYMENT', instructions='Use the QR Code at the entrance for mobile payments.')], 
                    how_to_arrive_instructions='Head west on Ruta 2. Take the exit at km 284 and follow the gravel road to Rio Claro Campgrounds.', 
                    properties=[Property(property_id='CampSite1', name='Camp Site 1', other_calendar_links=['https://booking.com/camp-site-1/calendar'], description='Secluded campsite near the river with a fire pit and picnic table.', amenities=['Fire Pit', 'Picnic Table', 'Portable Restroom'], price_per_night=20.0, currency='USD', max_guests=4, pick_up_keys_instructions='No keys required. The campsite is open. Please check-in at the reception upon arrival.')]), Business(business_id='f747b3a1-0408-4e3f-a9da-6333d42eadc2', business_name='Cielo Azul Resort', description='A luxury resort with a magnificent view of the mountains and countryside.', bnbot_id='@cielo_azul_resort', bnbot_configuration={'booking_deposit': 0.25, 'pre_book_time': 24.0}, location=Location(latitude=37.7749, longitude=-122.4194, address='Ruta 2 km 284', city='Mercedes', state='Soriano', country='Uruguay', postal_code='75000'), business_owners=[BusinessOwner(name='Elena Garcia', phone_number='+59892345678', email='elena.garcia@example.com')], payment_options=[PaymentOption(payment_method='CASH', instructions='Cash payments are accepted at the reception desk.')], how_to_arrive_instructions='Take the Ruta 2 highway and exit at km 284. Follow the signs for Cielo Azul Resort.', properties=[Property(property_id='VillaBella', name='Villa Bella', other_calendar_links=['https://booking.com/villa-bella/calendar'], description='Experience luxury in this exquisite villa with an infinity pool and a panoramic view of the mountains.', amenities=['Wi-Fi', 'Private Parking', 'Infinity Pool', 'Fitness Center'], price_per_night=500.0, currency='USD', max_guests=6, pick_up_keys_instructions='Visit the concierge at the main lobby to collect your keys.')]), 
            Business(business_id='ab0544c5-3029-4884-9354-d4da090c76d8', business_name='Complejo Enrique Joaquin', description='Complejo de campo muy bueno', bnbot_id='@complejo_enrique_joaquin', bnbot_configuration={'booking_deposit': 0.1, 'pre_book_time': 60.0}, location=Location(latitude=37.7749, longitude=-122.4194, address='Ruta 2 km 284', city='Mercedes', state='Soriano', country='Uruguay', postal_code='75000'), business_owners=[BusinessOwner(name='Gonzalo Sarasua', phone_number='+59899513718', email='sarasua.gonzalo@gmail.com')], payment_options=[PaymentOption(payment_method=None, instructions='Para realizar el deposito etc.')], how_to_arrive_instructions='El complejo queda en ruta 2 km 287, cerca del pejae. Yendo para Fray Bentos desde Mercedes a mano izquierda. Aqui esta la ubicacion en google maps: https://goo.gl/maps/R8gQZDHVXr2tiPQA8', properties=[Property(property_id='Sol', name='Cabaña "Sol"', other_calendar_links=['https://admin.booking.com/hotel/hoteladmin/ical.html?t=b20bbde6-86d6-4c7c-81d8-72c14ed4788c'], description='Impresionante villa con vistas panorámicas a las montañas. Esta lujosa propiedad ofrece un ambiente tranquilo y relajante con amplios espacios interiores y exteriores. Cuenta con una piscina privada, jardines exuberantes y una terraza para disfrutar de las maravillosas vistas. Perfecta para escapadas en familia o con amigos.', amenities=['Wi-Fi', 'private parking', 'pet-friendly', 'barbecue', 'private pool'], price_per_night=250.0, currency='USD', max_guests=8, pick_up_keys_instructions='Las llaves se encuentran en un box en la puerta de entrada. La clave para abrir el box es 12345.'), Property(property_id='Luna', name='Cabaña "Luna"', other_calendar_links=['https://admin.booking.com/hotel/hoteladmin/ical.html?t=8e422a9c-2ae6-4f83-8b28-18776ffcecfc'], description='Impresionante villa con vistas panorámicas a las montañas. Esta lujosa propiedad ofrece un ambiente tranquilo y relajante con amplios espacios interiores y exteriores. Cuenta con una piscina privada, jardines exuberantes y una terraza para disfrutar de las maravillosas vistas. Perfecta para escapadas en familia o con amigos.', amenities=['Wi-Fi', 'private parking', 'pet-friendly', 'barbecue', 'private pool'], price_per_night=120.0, currency='USD', max_guests=4, pick_up_keys_instructions='Las llaves se encuentran en un box en la puerta de entrada. La clave para abrir el box es 12345.')]), 
            Business(business_id='31ed8b67-1a7d-4895-bdcb-da4b0fc21f8e', business_name='La Estancia Verde', description='An exquisite countryside retreat offering a serene and natural atmosphere.', bnbot_id='@la_estancia_verde', bnbot_configuration={'booking_deposit': 0.15, 'pre_book_time': 72.0}, location=Location(latitude=37.7749, longitude=-122.4194, address='Ruta 2 km 284', city='Mercedes', state='Soriano', country='Uruguay', postal_code='75000'), business_owners=[BusinessOwner(name='Julia Rodriguez', phone_number='+59891234567', email='julia.rodriguez@example.com')], payment_options=[PaymentOption(payment_method='PAYPAL', instructions='Please use PayPal for transactions.')], how_to_arrive_instructions='Follow Ruta 2 until km 284. The retreat is located near the old mill.', properties=[Property(property_id='CasaGrande', name='Casa Grande', other_calendar_links=['https://booking.com/casa-grande/calendar'], description='A spacious villa with rustic interiors, private garden, and an outdoor pool.', amenities=['Wi-Fi', 'Private Parking', 'Pet-friendly', 'Outdoor Pool'], price_per_night=300.0, currency='USD', max_guests=10, pick_up_keys_instructions='Pick up the keys at the reception desk upon arrival.')]), Business(business_id='5e4c9d55-2af0-4875-ac86-d0232b0a8813', business_name='Campo del Sol', description='A family-friendly campsite and lodge in the heart of nature.', bnbot_id='@campo_del_sol', bnbot_configuration={'booking_deposit': 0.2, 'pre_book_time': 48.0}, location=Location(latitude=37.7749, longitude=-122.4194, address='Ruta 2 km 284', city='Mercedes', state='Soriano', country='Uruguay', postal_code='75000'), business_owners=[BusinessOwner(name='Carlos Mendez', phone_number='+59898765432', email='carlos.mendez@example.com')], payment_options=[PaymentOption(payment_method='CREDIT_CARD', instructions='We accept all major credit cards.')], how_to_arrive_instructions='Take exit 21 from Ruta 2, and follow the signs for Campo del Sol.', properties=[Property(property_id='LodgeOne', name='Lodge One', other_calendar_links=['https://booking.com/lodge-one/calendar'], description='A cozy lodge with a fireplace, perfect for couples or small families.', amenities=['Wi-Fi', 'Private Parking', 'Pet-friendly', 'Fireplace'], price_per_night=150.0, currency='USD', max_guests=4, pick_up_keys_instructions='The keys will be under the doormat. Please lock the door and return them under the doormat when checking out.')])
        ]
        for idx, test in enumerate(test_cases):
            print(f"Running test {idx}")
            mock_repository = MagicMock()
            mock_repository.list_all_businesses.return_value = businesses

            resolver = ListBusinessUseCase(mock_repository)

            # Act
            result = resolver.execute(test["load_business"])
            
            # Assert
            self.assertIsNotNone(result)
            # self.assertEqual(test["expected_resolver_done"], resolver.is_done())


if __name__ == '__main__':
    unittest.main()

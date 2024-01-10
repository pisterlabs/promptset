import os
import openai
import json
from dotenv import load_dotenv, find_dotenv
import time
import random
# from property.models import Property, PropertyFeature


# Se lee del archivo .env la api key de openai
_ = load_dotenv('.env')
openai.api_key = os.environ['openAI_api_key']

# Se carga la lista de propiedades de property_titles.json
with open('property_titles.json', 'r') as file:
    file_content = file.read()
    properties = json.loads(file_content)

# Se genera una función auxiliar que ayudará a la comunicación con la api de openai


def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]


# Definimos una instrucción general que le vamos a dar al modelo
instruction = "Vas a actuar como una persona dedicada al turismo, teniendo en cuenta cada caso. Describe cada propiedad ya sea apartamento y casa en menos de 100 palabras. La descripción debe incluir el tamaño, número de habitaciones y baños, y cualquier información adicional que sirva para crear un sistema de recomendación."

# Lista para almacenar todas las propiedades completas
all_properties = []

# Función para obtener el tipo de propiedad deacuerdo a la descripción


def get_type_of_property(description):
    if 'casa' in description.lower():
        return 'house'
    elif 'apartamento' in description.lower():
        return 'apartment'
    elif 'habitación' in description.lower() or 'habitacion' in description.lower() or 'habitaciones' in description.lower():
        return 'room'
    elif 'oficina' in description.lower():
        return 'office'
    elif 'aparta-estudio' in description.lower():
        return 'studio'
    else:
        return random.choice(['apartment', 'house', 'room', 'office', 'studio'])


# Función para obtener una ubicación random
def get_random_location():
    locations = ['El Centro', 'Juan Bautista Forero', 'Paraíso de Betel',
                 'El Prado', 'Regional', 'Las Tunas I', '16 de Julio']
    return f"{random.choice(locations)}, San Juan del Cesar, La Guajira"


# Función para generar direcciones random
def generate_random_address():
    streets = ["Calle", "Carrera", "Avenida", "Diagonal", "Transversal"]
    street_name = random.choice(streets)
    house_number = random.randint(1, 200)
    street_number = random.randint(1, 100)

    return f"{street_name} {street_number} - {house_number}"


# Función para generar una propiedad completa


def generate_complete_property(property_data):
    property_title = property_data['title']
    prompt = f"{instruction} Describe la propiedad \"{property_title}\"."
    description = get_completion(prompt).strip()

    property_data['description'] = description
    property_data['type_of_property'] = get_type_of_property(description)
    property_data['time_for_rent'] = random.choice(
        ['1', '3', '6', '12', '24', '36'])
    property_data['location'] = get_random_location()
    property_data['address'] = generate_random_address()
    property_data['size'] = round(random.uniform(50, 300), 2)
    property_data['rental_price'] = round(random.uniform(500000, 5000000), 2)
    property_data['status'] = random.choice(
        ['rented', 'available', 'pending', 'other'])

    property_data['property_feature'] = {
        "num_bedrooms": str(random.randint(1, 5)),
        "num_bathrooms": str(random.randint(1, 5)),
        "parking_spaces": random.randint(0, 3),
        "garden": random.choice([True, False]),
        "pool": random.choice([True, False]),
        "backyard": random.choice([True, False]),
        "furnished": random.choice([True, False]),
    }


# Itera sobre todas las propiedades
for i, property_data in enumerate(properties):
    generate_complete_property(property_data)
    all_properties.append(property_data)
    time.sleep(20)

# Guarda la lista de propiedades completas en un archivo JSON
file_path = "properties_complete.json"
with open(file_path, 'w') as json_file:
    json.dump(all_properties, json_file, indent=4)

print(f"Data saved to {file_path}")

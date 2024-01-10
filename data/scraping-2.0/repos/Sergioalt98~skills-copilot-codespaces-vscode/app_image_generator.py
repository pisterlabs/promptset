
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
import openai_secret_manager

# Configurar la conexión con el servicio de Computer Vision de Azure
endpoint = 'https://<nombre-del-servicio>.cognitiveservices.azure.com/'
subscription_key = '<clave-de-suscripción>'
computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

# Analizar una imagen existente
image_url = '<URL-de-la-imagen>'
image_analysis = computervision_client.analyze_image(image_url, visual_features=[VisualFeatureTypes.description])

# Obtener la descripción de la imagen
image_description = image_analysis.description.captions[0].text


# Generar una nueva imagen a partir de la descripción de texto

assert "openai" in openai_secret_manager.get_services()
secrets = openai_secret_manager.get_secret("openai")

generated_image = openai.Completion.create(
    engine="davinci",
    prompt=image_description,
    max_tokens=50,
    api_key=secrets["api_key"]
).choices[0].text

generated_image = openai.Completion.create(engine="davinci", prompt=image_description, max_tokens=50).choices[0].text




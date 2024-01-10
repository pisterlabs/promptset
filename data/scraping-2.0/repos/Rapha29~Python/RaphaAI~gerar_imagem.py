import openai as ia
from PIL import Image
import requests
from io import BytesIO

# GERAR A IMAGEM COM A IA
ia.api_key = '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'
request = input('Descreva a imagem a ser gerada: ')
response = ia.Image.create(
    prompt = request,
    n=1,
    size="1024x1024"
)
imagem_url = response['data'][0]['url']
print(f'URL da imagem gerada: \n{imagem_url}')

# MOSTRAR A IMAGEM A PARTIR DO LINK
try:
    response = requests.get(imagem_url)
    if response.status_code == 200:
        image_data = BytesIO(response.content)
        img = Image.open(image_data)
        img.show()
    else:
        print('Não foi possível baixar a imagem.')
except Exception as e:
    print(f'Erro ao abrir a imagem: {e}')


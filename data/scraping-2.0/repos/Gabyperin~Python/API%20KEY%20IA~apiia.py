import openai as ia

#Entra nesse site (https://platform.openai.com/account/api-keys) cria uma chave secreta, 
#copia e cola onde está pedindo.

ia.api_key = 'coloca aqui a api'
request = input('Descreva a imagem a ser gerada: ')
response = ia.Image.create(
    prompt = request,
    n=1,
    size="1024x1024"
)
imagem_url = response['data'][0]['url']
print(f'URL da imagem gerada: \n{imagem_url}')
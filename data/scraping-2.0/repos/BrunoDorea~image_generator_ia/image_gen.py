import openai
import requests


def create_images(text, nome):
    nome = nome + '.png'
    openai.api_key = "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"

    response = openai.Image.create(
    prompt = text,
    n = 1,
    size = "256x256"
    )
    image_url = response['data'][0]['url']

    image_response = requests.get(image_url)
    if image_response.status_code == 200:
        with open(f'imgs/{nome}', 'wb') as f:
            f.write(image_response.content)
        print('Imagem salva com sucesso.')
    else:
        print('Não foi possível baixar a imagem.')
        
    return f'imgs/{nome}'


if __name__ == "__main__":
    img = create_images("a white siamese cat", 'cat')
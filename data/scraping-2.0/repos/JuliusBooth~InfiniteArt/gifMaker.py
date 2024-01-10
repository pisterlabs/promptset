import openai
from config import OPEN_AI_KEY
import tempfile
from PIL import Image
import requests
from io import BytesIO
from firebase_helpers import upload_gif_to_firebase

client = openai.OpenAI(api_key=OPEN_AI_KEY)
IMAGE_SIZE = (512, 512)
IMAGE_SIZE_STR = f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}"

def get_image_from_url(url):
    response = requests.get(url)
    return resize_image(Image.open(BytesIO(response.content)))

def resize_image(img):
    return img.resize(IMAGE_SIZE)

def get_image_variation(image_url):
    img = get_image_from_url(image_url)
    with tempfile.NamedTemporaryFile(suffix='.png') as temp:
        img.save(temp, format='PNG')
        temp.flush()

        response = client.images.create_variation(
            image=open(temp.name, "rb"),
            n=1,
            size=IMAGE_SIZE_STR
        )
    image_url = response.data[0].url
    return image_url

def make_gif(num_frames, image_url):
    frame_urls = [image_url]
    for i in range(num_frames-1):
        image_url = get_image_variation(image_url)
        frame_urls.append(image_url)
    print(frame_urls)
    # frame_urls = ['https://firebasestorage.googleapis.com/v0/b/infiniteartglitch.appspot.com/o/images%2Fweaviate.jpeg?alt=media&token=916e89c5-f508-460e-bdb0-08cabf17d61a', 'https://oaidalleapiprodscus.blob.core.windows.net/private/org-XhlZGp1ng0N94tKZgzRAFXGE/user-T93mr5Zrj7Yvkm32Kff5riQD/img-wI3oSjXUk5GUF7524wNcKvJA.png?st=2023-11-28T23%3A10%3A27Z&se=2023-11-29T01%3A10%3A27Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-11-28T17%3A21%3A51Z&ske=2023-11-29T17%3A21%3A51Z&sks=b&skv=2021-08-06&sig=8OHic5L9tvO2/QejdD4FRXwSZyDdWwT2sUsQ0R9w4pc%3D', 'https://oaidalleapiprodscus.blob.core.windows.net/private/org-XhlZGp1ng0N94tKZgzRAFXGE/user-T93mr5Zrj7Yvkm32Kff5riQD/img-WWSI4lqo791Fg811tayTspYU.png?st=2023-11-28T23%3A10%3A37Z&se=2023-11-29T01%3A10%3A37Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-11-28T08%3A48%3A46Z&ske=2023-11-29T08%3A48%3A46Z&sks=b&skv=2021-08-06&sig=Zaohl0q4kdembp4jyctcPYtzJ6n0Owzqxt/8V3pvyiM%3D', 'https://oaidalleapiprodscus.blob.core.windows.net/private/org-XhlZGp1ng0N94tKZgzRAFXGE/user-T93mr5Zrj7Yvkm32Kff5riQD/img-Qk5zL2Lj0XzmxzA7B792h7Xg.png?st=2023-11-28T23%3A10%3A47Z&se=2023-11-29T01%3A10%3A47Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-11-28T13%3A23%3A43Z&ske=2023-11-29T13%3A23%3A43Z&sks=b&skv=2021-08-06&sig=doNCSYlxjzSMm29vWmVzFRoWtO5nSPf0C4VVDTZtcUY%3D']
    frames = [get_image_from_url(url) for url in frame_urls ]
    with tempfile.NamedTemporaryFile(delete=False, suffix='.gif', mode='w+b') as temp:
        frames[0].save(temp, save_all=True, append_images=frames[1:], loop=0, duration=300, optimize=True)
        gif_url = upload_gif_to_firebase(temp.name)
    return gif_url

    

if __name__ == '__main__':
    
    image_url = "https://firebasestorage.googleapis.com/v0/b/infiniteartglitch.appspot.com/o/images%2Fweaviate.jpeg?alt=media&token=916e89c5-f508-460e-bdb0-08cabf17d61a"
    variant = make_gif(4,image_url)
    print(variant)
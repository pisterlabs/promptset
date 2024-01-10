from django.db.models.signals import post_save
from django.dispatch import receiver
from django.conf import settings
from .models import Chef
from openai import OpenAI
import requests
from django.core.files.base import ContentFile

@receiver(post_save, sender=Chef)
def create_chef_image(sender, instance, created, **kwargs):
    if created and not instance.profile_pic:
        # Logic to call DALL-E API and save the image
        image_url = generate_chef_image()  # Function to call DALL-E API
        response = requests.get(image_url)
        if response.status_code == 200:
            image_name = f'{instance.user.username}_chef_placeholder.png'
            instance.profile_pic.save(image_name, ContentFile(response.content), save=True)

def generate_chef_image():
    client = OpenAI(api_key=settings.OPENAI_KEY)  # Initialize with your API credentials
    prompt = "A gender-neutral chef in a professional kitchen with their back to the camera as if they're preparing a dish."
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    image_url = response.data[0].url
    return image_url

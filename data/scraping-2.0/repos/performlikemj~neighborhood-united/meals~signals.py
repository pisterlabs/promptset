# meals/signals.py
from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import Meal
import requests
from django.core.files.base import ContentFile
from openai import OpenAI
from django.conf import settings

@receiver(post_save, sender=Meal)
def create_meal_image(sender, instance, created, **kwargs):
    if created and not instance.image:
        # Generate image using DALL-E 3
        client = OpenAI(api_key=settings.OPENAI_KEY)
        prompt = f"A delicious meal: {instance.name}. {instance.description}"
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url

        # Download the image
        response = requests.get(image_url)
        if response.status_code == 200:
            image_name = f'{instance.name}_meal_image.png'
            instance.image.save(image_name, ContentFile(response.content), save=True)

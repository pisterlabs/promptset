from django.core.management.base import BaseCommand
from django.core.files.base import ContentFile
from meals.models import Meal
from openai import OpenAI
import requests
from django.conf import settings

class Command(BaseCommand):
    help = 'Generate and assign DALL-E images to meals without images'

    def add_arguments(self, parser):
        parser.add_argument('--force', action='store_true', help='Force regenerate images for all meals')

    def handle(self, *args, **options):
        force_regenerate = options['force']
        meals = Meal.objects.all()
        client = OpenAI(api_key=settings.OPENAI_KEY)

        for meal in meals:
            if meal.image and not force_regenerate:
                continue  # Skip meals that already have images, unless force regenerate

            prompt = f"{meal.name}"
            if meal.description:
                prompt += f" - {meal.description}"

            try:
                response = client.images.generate(
                    model="dall-e-3",
                    prompt=prompt,
                    size="1024x1024",
                    quality="standard",
                    n=1,
                )
                image_url = response.data[0].url
                image_content = requests.get(image_url).content
                meal.image.save(f"{meal.name}_image.png", ContentFile(image_content))
                self.stdout.write(self.style.SUCCESS(f'Image updated for meal: {meal.name}'))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Failed to update image for meal: {meal.name}. Error: {e}'))

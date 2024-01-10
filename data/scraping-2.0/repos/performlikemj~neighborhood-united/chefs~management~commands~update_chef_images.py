import requests
from django.core.files.base import ContentFile
from django.core.management.base import BaseCommand
from chefs.models import Chef
from openai import OpenAI
from django.conf import settings

class Command(BaseCommand):
    help = 'Generates images for chefs without a profile picture'

    def download_image(self, url):
        response = requests.get(url)
        if response.status_code == 200:
            return ContentFile(response.content)
        else:
            raise Exception(f"Failed to download image: {response.status_code}")

    def handle(self, *args, **kwargs):
        chefs_without_images = Chef.objects.filter(profile_pic='')

        for chef in chefs_without_images:
            try:
                image_url = self.generate_chef_image()  # Use the function to generate image URL
                image_file = self.download_image(image_url)
                chef.profile_pic.save(f'chef_{chef.id}.png', image_file)
                chef.save()
                self.stdout.write(self.style.SUCCESS(f'Successfully updated image for chef {chef.user.username}'))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Failed to update image for chef {chef.user.username}: {e}'))

    def generate_chef_image(self):
        # Your logic to call DALL-E API and get the image URL
        client = OpenAI(api_key=settings.OPENAI_KEY)
        prompt = "A cartoon-like gender-neutral chef in a professional kitchen with their back to the camera as if they're preparing a dish."
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        return image_url

from django.core.management.base import BaseCommand

from main.utils import OpenAIImageHelper, ImageHelper


class Command(BaseCommand):
    help = 'gen_img'

    def add_arguments(self, parser):
        parser.add_argument(dest='prompt', help='prompt')

    def handle(self, *args, **options):
        print('gen_img')
        prompt = options['prompt']
        print(prompt)

        # urls = OpenAIImageHelper.gen_img(prompt, size="256x256")
        # print(urls)
        # ImageHelper.imgurl2file(urls[0], "local/gen_img.png")
        with open("local/gen_img.png", "rb") as f:
            d = f.read()
            urls = OpenAIImageHelper.variate_img(d, size="256x256")
            ImageHelper.imgurl2file(urls[0], "local/variate_img.png")

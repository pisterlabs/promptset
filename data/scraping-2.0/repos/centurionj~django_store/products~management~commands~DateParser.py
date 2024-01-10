import random
import time

from django.conf import settings
from django.db import transaction

import requests
from bs4 import BeautifulSoup
import openai
from faker import Faker

from products.models import Brand, Category, Product, Size, ProductImage


class DateParser:
    """
        parsing товаров через requests и BeautifulSoup
    """

    fake = Faker('ru_RU')

    def __init__(self, url):
        self.url = url

    def query_gpt(self, prompt):
        openai.api_key = settings.GPT_API_KEY
        response = openai.Completion.create(
            engine='text-davinci-003',
            prompt=prompt,
            max_tokens=2048,
            n=1,
            stop=None,
            temperature=0.7
        )
        return response['choices'][0]['text'].strip() if response.get('choices') else None

    def collect_data(self):
        gender_list = ["M", "F"]
        brand_list = ["Nike", "Chanel", "Gucci", "Adidas", "ТВОЁ"]
        r = requests.get(self.url)
        soup = BeautifulSoup(r.content, 'lxml')
        all_div = soup.find_all('div', class_='product-list__product-wrapper')

        for item in all_div:
            time.sleep(21)
            product_name = item.find('span', class_='product__title-text').text.strip()
            image_link = item.find('img', class_='product__image').get('src')
            image_link_2 = item.find('img', class_='product__image v-cloak--hidden').get('src')
            prompt = (
                f"Какая это категория товара {product_name}, "
                f"в ответе только конкретная категория товара одним словом, "
                f"а не общее 'одежда', например шорты, майки и тд"
            )
            category = str(self.query_gpt(prompt=prompt)).replace('.', '')
            current_price = float(item.find('span', class_='product__price-value').text.replace('\xa0', ''))
            brand = random.choice(brand_list)
            description = self.fake.paragraph(nb_sentences=7)
            gender = random.choice(gender_list)

            with transaction.atomic():
                brand_obj, _ = Brand.objects.get_or_create(name=brand)
                category_obj, _ = Category.objects.get_or_create(name=category)

                brand_obj.categories.add(category_obj)

                product, _ = Product.objects.get_or_create(
                    name=product_name,
                    description=description,
                    price=current_price,
                    category=category_obj,
                    brand=brand_obj,
                    image=image_link,
                    gender=gender,
                )

                size_list = ['XS', 'S', 'M', 'L', 'XL', 'XXL']
                for size_title in size_list:
                    size_obj, _ = Size.objects.get_or_create(title=size_title)
                    product.sizes.add(size_obj)

                product_image, _ = ProductImage.objects.get_or_create(product=product, image=image_link_2)

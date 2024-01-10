import json
import logging
import openai
from bs4 import BeautifulSoup
from src import editing


def classify_categories(prestashop, openai_conn, product_ids_list: list[int]):
    openai.api_key = openai_conn

    with open('data/cats_dict.json', encoding='utf-8') as file:
        cats_classify = json.load(file).get('cats_classify')
        cats_id_dict = json.load(file).get('cat_id')

    for product_id in product_ids_list:
        product = prestashop.get('products', product_id).get('product')
        product_desc = product['description_short']['language']['value']
        product_cats = []

        with open('data/prompts/classify_product.txt', 'r', encoding='utf-8') as file:
            prompt_template = file.read().strip()
        prompt = prompt_template.format(product=product_desc, cats=cats_classify)

        response = openai.Completion.create(engine='text-davinci-003', prompt=prompt, max_tokens=400, temperature=0.2)
        generated_text = response.choices[0].text

        for part in generated_text.split(","):
            category_name = part.strip()
            if category_name in list(cats_classify.values()):
                product_cats.append(category_name)

        product_cats_ids = ['2'] + [cats_id_dict[cat] for cat in product_cats]
        product_cats_upload = [{'id': cat_id} for cat_id in product_cats_ids]

        product['id_category_default'] = product_cats_ids[-1]
        product['associations']['categories']['category'] = product_cats_upload

        editing.edit_presta_product(prestashop, product=product)
    logging.info('FINISHED product classification')


def write_descriptions(prestashop, openai_conn, product_ids_list: list[int]):
    openai.api_key = openai_conn

    for product_id in product_ids_list:
        product = prestashop.get('products', product_id).get('product')
        product_name = product['name']['language']['value']
        product_desc = product['description']['language']['value']
        product_summary, product_ingredients = editing.manipulate_desc(product_desc)

        with open('data/prompts/write_desc_2.txt', 'r', encoding='utf-8') as file:
            prompt_template = file.read().strip()
        prompt = prompt_template.format(product_name=product_name, product_desc=product_summary)
        response = openai.Completion.create(engine='text-davinci-003', prompt=prompt, max_tokens=1900, temperature=0.25)

        desc_short, desc_long = editing.make_desc(response.choices[0].text.strip())

        with open('data/prompts/write_active.txt', 'r', encoding='utf-8') as file:
            prompt_template = file.read().strip()
        prompt = prompt_template.format(product_desc=product_ingredients)
        response = openai.Completion.create(engine='text-davinci-003', prompt=prompt, max_tokens=1500, temperature=0.25)

        desc_active = editing.make_active(response.choices[0].text.strip())

        product['description_short']['language']['value'] = desc_short
        product['description']['language']['value'] = desc_long + desc_active

        editing.edit_presta_product(prestashop, product=product)
    logging.info('FINISHED writing product descriptions')


def write_meta(prestashop, openai_conn, product_ids_list: list[int]):
    openai.api_key = openai_conn

    for product_id in product_ids_list:
        product = prestashop.get('products', product_id)['product']
        product_name = product['name']['language']['value']
        product_desc = product['description_short']['language']['value']
        product_desc = BeautifulSoup(product_desc, 'html.parser').get_text()

        with open('data/prompts/write_meta_2.txt', 'r', encoding='utf-8') as file:
            prompt_template = file.read().strip()
        prompt = prompt_template.format(product_name=product_name, product_desc=product_desc)
        response = openai.Completion.create(engine='text-davinci-003', prompt=prompt, max_tokens=400, temperature=0.3)

        text = response.choices[0].text.strip()

        meta_title = text.split('META DESCRIPTION:')[0].split('META TITLE:')[1].strip()
        meta_desc = editing.truncate_meta(text.split('META DESCRIPTION:')[1].strip())

        product['meta_title']['language']['value'] = meta_title
        product['meta_description']['language']['value'] = meta_desc

        editing.edit_presta_product(prestashop, product=product)
    logging.info('FINISHED writing meta descriptions')


def apply_ai_actions(prestashop, openai_conn, product_ids: list[int],
                     classify_ai: bool = 0, descriptions_ai: bool = 0, meta_ai: bool = 0, inci_unit: bool = 0):

    if classify_ai:
        classify_categories(prestashop, openai_conn, product_ids)
    if descriptions_ai:
        write_descriptions(prestashop, openai_conn, product_ids)
    if meta_ai:
        write_meta(prestashop, openai_conn, product_ids)
    if inci_unit:
        editing.fill_inci(prestashop, product_ids=product_ids, source='aleja')
        editing.set_unit_price_api_sql(prestashop, product_ids=product_ids)

    logging.info('Finished all AI actions.')

import openai
import os
import json
from wp_content_uploader import ContentUploader

from amazon_product_scraper import AmazonProductScraper

openai.api_key = os.environ.get('OPENAI_API_KEY')
SITE_URL = os.environ.get("SITE_URL")
WP_USERNAME = os.environ.get("WP_USERNAME")
WP_PASSWORD = os.environ.get("WP_PASSWORD")
USERNAME = os.environ.get("USERNAME")
PASSWORD = os.environ.get("PASSWORD")
CHROME_DRIVER_PATH = os.environ.get("CHROME_DRIVER_PATH")


def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


def create_path_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created path: {path}")
    else:
        print(f"Path already exists: {path}")


class StoreContentGenerator:
    def __init__(self, store, products, content_size=2):
        self.store = store
        self.products = products
        self.content = {}
        self.content_size = content_size
        self.content_uploader = ContentUploader(WP_USERNAME, WP_PASSWORD, SITE_URL)
        self.checkpoint = {
                            "start_content_structure": 0,
                            "home_page_set": 0,
                            "set_categories": 0,
                            "set_subcategories": {
                                "section_idx": 0,
                                "category_idx": 0,
                                "idx": 0
                            },
                            "set_products": {
                                "category_idx": 0,
                                "subcategory_idx": 0,
                                "idx": 0
                            },
                            "set_products_articles": {
                                "category_idx": 0,
                                "subcategory_idx": 0,
                                "idx": 0
                            },
                            "set_blog": {
                                "category_idx": 0,
                                "subcategory_idx": 0,
                                "idx": 0
                            },
                            "set_blog_articles": {
                                "category_idx": 0,
                                "subcategory_idx": 0,
                                "topic_idx": 0,
                                "idx": 0
                            },
                            "set_product_reviews": {
                                "category_idx": 0,
                                "subcategory_idx": 0,
                                "product_idx": 0,
                                "idx": 0,
                                "set": 0
                            }
                        }

    def start_content_structure(self):
        print("\n--------INITIAL STRUCTURE SET-------")
        if not self.checkpoint["start_content_structure"]:
            main_menu_prompt = f"""
            Instrucciones:
            Escribe como un experto SEO.
    
            Este es un diccionario que representa un listado de las secciones que debe tener el menu principal de una {self.store}. Tu objetivo es proporcionar la estructura del diccionario en formato JSON. solo el JSON
    
            - El diccionario principal se llama "menu".
            - Cada nombre de sección debe ser una clave dentro del diccionario principal.
            - Para cada sección, incluye las siguientes claves:
                - "titulo":  nombre de la seccion.
                - "descripcion": nombre completo relacionado con {self.store}.
                - "categorizable" : 1 si es categorizable y 0 si no.
            """

            main_menu_response = get_completion(main_menu_prompt)
            main_menu_dict = json.loads(main_menu_response)

            if main_menu_dict['menu'].get('blog'):
                main_menu_dict['menu']['blog']['categorizable'] = 1

            self.content = main_menu_dict
            self.checkpoint["start_content_structure"] = 1
            self.save_checkpoint()
            self.save_content()
        print("Completed ✓\n")

    def set_categories(self):
        print("\n-------CATEGORIES SET-------")

        if not self.checkpoint["set_categories"]:

            idx = 0
            for section, characteristics in self.content['menu'].items():

                if characteristics['categorizable']:

                    categories_promt = f"""
                    Escribe como un Experto SEO
                    {self.content_size} categorias relevantes para la seccion de {section} de una {self.store}.
                    En formato de lista de python. solo la lista de python.
                    siguiendo el siguiente formato: [categoria1, categoria2]
                    """
                    categories_response = get_completion(categories_promt)
                    categories_list = eval(categories_response)
                    characteristics["categorias"] = []
                    for category in categories_list:
                        characteristics["categorias"].append({
                            "nombre": category
                        })
                print(idx+1, "category set")
                idx += 1
            self.checkpoint["set_categories"] = 1
            self.save_content()
            self.save_checkpoint()

        print("Completed ✓\n")

    def set_subcategories(self):
        print("\n-------SUBCATEGORIES SET-------")
        section_idx = self.checkpoint["set_subcategories"]["section_idx"]
        category_idx = self.checkpoint["set_subcategories"]["category_idx"]
        idx = self.checkpoint["set_subcategories"]["idx"]

        for section, characteristics in list(self.content['menu'].items())[section_idx:]:
            if characteristics['categorizable']:
                characteristics["category_id"] = self.content_uploader.new_category(characteristics)
                for category in characteristics["categorias"][category_idx:]:
                    category["parent_id"] = characteristics["category_id"]
                    category["category_id"] = self.content_uploader.new_subcategory(category)
                    subcategories_prompt = f"""
                    Escribe como un Experto SEO
                    {self.content_size} Subcategorias relevantes para la categoria de "{category["nombre"]}" de la seccion de "{section}" de una {self.store}.
                    En formato de lista de python. solo la lista de python.
                    siguiendo el siguiente formato: [subcategoria1, subcategoria2]
                    """
                    subcategories_response = get_completion(subcategories_prompt)
                    subcategories_list = eval(subcategories_response)
                    category["subcategorias"] = []
                    for subcategory in subcategories_list:
                        new_subcategory = {
                            "nombre": subcategory,
                            "parent_id": category["category_id"]
                        }
                        new_subcategory["category_id"] = self.content_uploader.new_subcategory(new_subcategory)
                        category["subcategorias"].append(new_subcategory)
                    print(idx+1, "subcategory set")
                    idx += 1
                    category_idx += 1
                    self.checkpoint["set_subcategories"]["category_idx"] = category_idx
                    self.checkpoint["set_subcategories"]["idx"] = idx
                    self.save_checkpoint()
                    self.save_content()
                category_idx = 0
                self.checkpoint["set_subcategories"]["category_idx"] = category_idx
                self.save_checkpoint()

            section_idx += 1
            self.checkpoint["set_subcategories"]["section_idx"] = section_idx
            self.save_checkpoint()
        print("Completed ✓\n")

    def set_products(self):
        print("\n-------PRODUCTS SET-------")
        category_idx = self.checkpoint["set_products"]["category_idx"]
        subcategory_idx = self.checkpoint["set_products"]["subcategory_idx"]
        idx = self.checkpoint["set_products"]["idx"]

        if self.content['menu']['productos']:

            for category in self.content['menu']['productos']['categorias'][category_idx:]:
                for subcategory in category['subcategorias'][subcategory_idx:]:
                    products_prompt = f"""
                    Escribe como un Experto en la Busqueda de {self.products} en Amazon.
                    {self.content_size*2} productos relevantes para la subcategoria de "{subcategory["nombre"]}" de la categoria de "{category["nombre"]}" de  la seccion de "productos"  de una {self.store}. 
                    
                    Asegurate de que sean productos que puedan encontrarse en Amazon.
                    
                    En formato de lista de python. solo la lista de python.
                    siguiendo el siguiente formato: ["producto1", "producto2"]
                    """
                    products_response = get_completion(products_prompt)
                    products_list = eval(products_response)
                    subcategory["productos"] = []
                    for product in products_list:

                        subcategory["productos"].append({
                            "nombre": product,
                        })
                    print(idx+1, "subcategory products set")
                    idx += 1
                    subcategory_idx += 1
                    self.checkpoint["set_products"]["subcategory_idx"] = subcategory_idx
                    self.checkpoint["set_products"]["idx"] = idx
                    self.save_checkpoint()
                    self.save_content()
                subcategory_idx = 0
                self.checkpoint["set_products"]["subcategory_idx"] = subcategory_idx
                self.save_checkpoint()

                category_idx += 1
                self.checkpoint["set_products"]["category_idx"] = category_idx
                self.save_checkpoint()

        print("Completed ✓\n")

    def set_product_reviews(self):
        print("\n-------PRODUCT REVIEWS SET-------")
        category_idx = self.checkpoint["set_product_reviews"]["category_idx"]
        subcategory_idx = self.checkpoint["set_product_reviews"]["subcategory_idx"]
        product_idx = self.checkpoint["set_product_reviews"]["product_idx"]
        idx = self.checkpoint["set_product_reviews"]["idx"]

        if self.content['menu']['productos'] and not self.checkpoint["set_product_reviews"]["set"]:
            scraper = AmazonProductScraper(CHROME_DRIVER_PATH, USERNAME, PASSWORD)
            scraper.login()
            for category in self.content['menu']['productos']['categorias'][category_idx:]:
                for subcategory in category['subcategorias'][subcategory_idx:]:
                    for product in subcategory["productos"][product_idx:]:
                        download_image_path = f"product_images/{category['nombre']}/{subcategory['nombre']}/"
                        create_path_if_not_exists(download_image_path)
                        title, ref_url, image_path = scraper.search_product(product["nombre"], download_image_path)
                        product["titulo"] = title
                        product["ref_url"] = ref_url
                        product["image_path"] = image_path
                        scraper.process_images(download_image_path)
                        if product["titulo"]:
                            review_prompt = """
                                            Escribe como un Experto en Ventas Online.
                                            una reseña relevante para el producto "%s" de la subcategoria de "%s" de la categoria de "%s" de la seccion de "productos" de una %s (2100 palabras). 
                                            En formato JSON. solo el JSON.
                                             siguiendo el siguiente formato: 
                                            {
                                            "titulo": titulo,
                                            "meta-descripcion": meta descripcion,
                                            "contenido": contenido de la reseña en formato HTML,
                                            "llamada a la accion": llamada a la accion con boton de comprar con la url: "%s" en formato HTML
                                            }
                                            """ % (product["titulo"], subcategory["nombre"], category["nombre"], self.store, product["ref_url"])
                            review_response = get_completion(review_prompt)
                            review = json.loads(review_response)
                            product["reseña"] = review
                        if product["image_path"]:

                            product["image_id"] = self.content_uploader.upload_image(product["image_path"])
                            self.content_uploader.new_product_post(product, subcategory["category_id"])
                        print(idx + 1, "product review set")
                        idx += 1
                        product_idx += 1
                        self.checkpoint["set_product_reviews"]["idx"] = idx
                        self.checkpoint["set_product_reviews"]["product_idx"] = product_idx
                        self.save_checkpoint()
                        self.save_content()
                    product_idx = 0
                    self.checkpoint["set_product_reviews"]["product_idx"] = product_idx
                    self.save_checkpoint()
                    subcategory_idx += 1
                    self.checkpoint["set_product_reviews"]["subcategory_idx"] = subcategory_idx
                    self.checkpoint["set_product_reviews"]["idx"] = idx
                    self.save_checkpoint()
                subcategory_idx = 0
                self.checkpoint["set_product_reviews"]["subcategory_idx"] = subcategory_idx
                self.save_checkpoint()

                category_idx += 1
                self.checkpoint["set_product_reviews"]["category_idx"] = category_idx
                self.save_checkpoint()
        self.checkpoint["set_product_reviews"]["set"] = 1
        self.save_checkpoint()
        print("Completed ✓\n")

    def set_products_articles(self):
        print("\n-------PRODUCTS ARTICLES SET-------")
        category_idx = self.checkpoint["set_products_articles"]["category_idx"]
        subcategory_idx = self.checkpoint["set_products_articles"]["subcategory_idx"]
        idx = self.checkpoint["set_products_articles"]["idx"]

        if self.content['menu']['productos']:

            for category in self.content['menu']['productos']['categorias'][category_idx:]:
                for subcategory in category['subcategorias'][subcategory_idx:]:
                    article_prompt = """
                    Escribe como un Experto SEO.
                    un articulo relevante para la subcategoria de "%s" de la categoria de "%s" de  la seccion de "productos" de una %s (2100 palabras). 

                    En formato JSON. solo el JSON.
                     siguiendo el siguiente formato: 
                    {
                    "titulo": nombre del titulo,
                    "meta-descripcion": meta descripcion,
                    "ventajas": 7 ventajas desarrolladas de consumir "%s". en formato HTML,
                    "preguntas-frecuentes": 7 preguntas frecuentes con sus respuestas antes de comprar un producto de "%s" en formato HTML,
                    }
                    """ % (subcategory["nombre"], category["nombre"], self.store, subcategory["nombre"], subcategory["nombre"])
                    article_response = get_completion(article_prompt)
                    article = json.loads(article_response)
                    subcategory["articulo"] = article
                    self.content_uploader.new_page_with_gallery(subcategory["nombre"], subcategory["productos"], subcategory["articulo"], subcategory["parent_id"])
                    print(idx + 1, "products article set")
                    idx += 1
                    subcategory_idx += 1
                    self.checkpoint["set_products_articles"]["subcategory_idx"] = subcategory_idx
                    self.checkpoint["set_products_articles"]["idx"] = idx
                    self.save_checkpoint()
                    self.save_content()
                subcategory_idx = 0
                self.checkpoint["set_products_articles"]["subcategory_idx"] = subcategory_idx
                self.save_checkpoint()

                category_idx += 1
                self.checkpoint["set_products_articles"]["category_idx"] = category_idx
                self.save_checkpoint()

        print("Completed ✓\n")



    def set_blog(self):
        print("\n-------BLOG SET-------")
        category_idx = self.checkpoint["set_blog"]["category_idx"]
        subcategory_idx = self.checkpoint["set_blog"]["subcategory_idx"]
        idx = self.checkpoint["set_blog"]["idx"]

        if self.content['menu']['blog']:
            for category in self.content['menu']['blog']['categorias'][category_idx:]:
                for subcategory in category['subcategorias'][subcategory_idx:]:
                    topics_prompt = f"""
                            Escribe como un Experto en SEO.
                            {self.content_size} temas relevantes para la subcategoria de "{subcategory["nombre"]}" de la categoria de "{category["nombre"]}" de  la seccion de "blog"  de una {self.store}. 
                            En formato de lista de python. solo la lista de python.
                            siguiendo el siguiente formato: [tema1, tema2]
                            """
                    topics_response = get_completion(topics_prompt)
                    topics_list = eval(topics_response)
                    subcategory["temas"] = []
                    for topic in topics_list:
                        subcategory["temas"].append({
                            "nombre": topic
                        })
                    print(idx + 1, "subcategory topics set")
                    idx += 1
                    subcategory_idx += 1
                    self.checkpoint["set_blog"]["subcategory_idx"] = subcategory_idx
                    self.checkpoint["set_blog"]["idx"] = idx
                    self.save_checkpoint()
                    self.save_content()
                subcategory_idx = 0
                self.checkpoint["set_blog"]["subcategory_idx"] = subcategory_idx
                self.save_checkpoint()

                category_idx += 1
                self.checkpoint["set_blog"]["category_idx"] = category_idx
                self.save_checkpoint()

        print("Completed ✓\n")

    def set_blog_articles(self):
        print("\n-------BLOG ARTICLES SET-------")
        category_idx = self.checkpoint["set_blog_articles"]["category_idx"]
        subcategory_idx = self.checkpoint["set_blog_articles"]["subcategory_idx"]
        topic_idx = self.checkpoint["set_blog_articles"]["topic_idx"]
        idx = self.checkpoint["set_blog_articles"]["idx"]

        if self.content['menu']['blog']:
            scraper = AmazonProductScraper(CHROME_DRIVER_PATH, USERNAME, PASSWORD)
            for category in self.content['menu']['blog']['categorias'][category_idx:]:
                for subcategory in category['subcategorias'][subcategory_idx:]:
                    for topic in subcategory["temas"][topic_idx:]:
                        download_image_path = f"post_images/{category['nombre']}/{subcategory['nombre']}/"
                        create_path_if_not_exists(download_image_path)
                        image_path = scraper.get_post_image(topic["nombre"], download_image_path)
                        topic["image_path"] = image_path
                        scraper.process_images(download_image_path)
                        article_prompt = """
                                        Escribe como un Experto en SEO.
                                        un articulo relevante para el tema de "%s" de la subcategoria de "%s" de la categoria de "%s" de la seccion de "blog" de una %s (2100 palabras). 
                                        En formato JSON. solo el JSON.
                                         siguiendo el siguiente formato: 
                                        {
                                        "titulo": nombre del titulo,
                                        "meta-descripcion": meta descripcion,
                                        "contenido": contenido del articulo en formato HTML,
                                        }
                                        """ % (topic["nombre"], subcategory["nombre"], category["nombre"], self.store)
                        article_response = get_completion(article_prompt)
                        article = json.loads(article_response)
                        topic["articulo"] = article
                        if topic["image_path"]:
                            topic["image_id"] = self.content_uploader.upload_image(topic["image_path"])
                            self.content_uploader.new_blog_post(topic, subcategory["category_id"])
                        print(idx + 1, "topic article set")
                        idx += 1
                        topic_idx += 1
                        self.checkpoint["set_blog_articles"]["idx"] = idx
                        self.checkpoint["set_blog_articles"]["topic_idx"] = topic_idx
                        self.save_checkpoint()
                        self.save_content()
                    topic_idx = 0
                    self.checkpoint["set_blog_articles"]["topic_idx"] = topic_idx
                    self.save_checkpoint()
                    subcategory_idx += 1
                    self.checkpoint["set_blog_articles"]["subcategory_idx"] = subcategory_idx
                    self.save_checkpoint()
                subcategory_idx = 0
                self.checkpoint["set_blog_articles"]["subcategory_idx"] = subcategory_idx
                self.save_checkpoint()
                category_idx += 1
                self.checkpoint["set_blog_articles"]["category_idx"] = category_idx
                self.save_checkpoint()

        print("Completed ✓\n")


    def set_homepage(self):
        print("\n-------HOMEPAGE SET-------")
        if not self.checkpoint["home_page_set"]:
            homepage_prompt = """
            Escribe como un Experto SEO
            un articulo relevante e interesante sobre %s en general (3500 palabras).
            En formato JSON. solo el JSON.
             siguiendo el siguiente formato: 
            {
            "titulo": nombre del titulo,
            "meta-descripcion": meta descripcion,
            "contenido": contenido del articulo en formato HTML,
            }
            """ % self.products

            homepage_response = get_completion(homepage_prompt)
            homepage_dict = json.loads(homepage_response)
            self.content["menu"]["inicio"]["articulo"] = homepage_dict
            self.content_uploader.new_page(self.content["menu"]["inicio"]["articulo"])
            self.checkpoint["home_page_set"] = 1
            self.save_checkpoint()
            self.save_content()
        print("Completed ✓\n")

    def get_current_content(self):
        return self.content

    def load_checkpoint(self):
        try:
            with open('checkpoint.json', 'r') as file:
                self.checkpoint = json.load(file)
        except FileNotFoundError:
            self.save_checkpoint()

    def save_checkpoint(self):
        with open('checkpoint.json', 'w') as file:
            json.dump(self.checkpoint, file)

    def load_content(self):
        try:
            with open('content.json', 'r') as file:
                self.content = json.load(file)
        except FileNotFoundError:
            self.save_content()

    def save_content(self):
        with open('content.json', 'w') as file:
            json.dump(self.content, file)


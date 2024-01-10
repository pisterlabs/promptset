import erppeek
import openai
import pdb

openai.api_key = "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"


def connection_odoo_local():
    """Connect to Odoo local"""
    odoo_url = "http://localhost:8069"
    odoo_db = "odoo_ai_16"
    odoo_username = "admin"
    odoo_password = "admin"
    return erppeek.Client(odoo_url, odoo_db, odoo_username, odoo_password)


def get_products(client):
    """Récupération des produits"""
    products = client.model("product.template").search([])
    product_list = []

    for product in products:
        product = client.model("product.template").browse(product)
        product_list.append(product)
    return product_list


def main():
    # connection au odoo local
    odoo = connection_odoo_local()

    # Récupération des produits
    products_list = get_products(odoo)

    for product in products_list:
        print(product.description)
        description = product.description
        response = openai.Completion.create(
            engine="text-davinci-003",  # Vous pouvez utiliser le moteur GPT-3.5
            prompt="Traduire en chinois : " + str(description),
            max_tokens=60,  # Ajustez selon la longueur attendue de la réponse
        )
        product_odoo = odoo.model("product.template").browse(
            [("name", "=", product.name)]
        )
        vals = {
            "description": response.choices[0].text,
        }
        if product_odoo:
            product_odoo.write(vals)

        print(response.choices[0].text)


if __name__ == "__main__":
    main()

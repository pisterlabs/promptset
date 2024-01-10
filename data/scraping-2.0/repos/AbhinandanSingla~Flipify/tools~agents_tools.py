from langchain.tools import tool

from flipkart_scraper.product import Product
from flipkart_scraper.scrapper import search_products, dataframe_to_object


@tool
def searchCloth(query: str) -> str:
    """This function searches the products like cloths,footwear,outfits,gifts/colours/size/shades etc."""
    print('\n')
    print(query)
    searched_products = []
    df = search_products(query)
    for _, row in df.iterrows():
        searched_products.append(dataframe_to_object(Product, row))
    print(len(searched_products))

    return f"""
     First Part: Following is the product information You should display the first product to the customer, and interact with him in a convincing way providing information about the customer ratings and product quality. Make the conversation appear human-like (important).
    Second Part:
    Now, the next step is to essentially display the corresponding link to relevant product as per the following:
    {[{'name': searched_products[x].name, 'price': searched_products[x].price,
       'image_link': searched_products[x].image, 'product_link': searched_products[x].detail_url} for x in
      range(len(searched_products))][:3]}
     As soon as a relevant product is found, stop calling the function again.

"""


if __name__ == '__main__':
    print(search_products("shirts"))

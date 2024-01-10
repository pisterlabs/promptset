import os
import openai


def generate_code():

    openai.api_key = os.getenv("OPENAI_API_KEY")

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="provide me a complete file structure for a nextjs ecommerce app in form of string\n\n\n{\n  \"root_directory\": {\n    \"client\": {\n      \"pages\": [\n        \"/\",\n        \"/products\",\n        \"/cart\",\n        \"/checkout\"\n      ],\n      \"public\": {\n        \"assets\": {\n          \"images\": [],\n          \"files\": []\n        }\n      },\n      \"components\": {\n        \"navbar\": {\n          \"Header.js\",\n          \"Nav.js\",\n          \"Footer.js\"\n        }, \n        \"product\": {\n          \"Product.js\",\n          \"ProductList.js\"\n        }, \n        \"checkout\": {\n          \"CheckoutForm.js\", \n          \"PaymentForm.js\" \n        }\n      }\n    },\n    \"server\": {\n      \"apis\": {\n        \"products\": {\n          \"ProductController.js\"\n        }, \n        \"cart\": {\n          \"CartController.js\"\n        }, \n        \"checkout\": {\n          \"CheckoutController.js\"\n        }\n      }\n    },\n    \".env\":\n\n{\n    \"ecommerceApp\": {\n        \"package.json\": {},\n        \"next-env.d.ts\": {},\n        \"src\": {\n            \"components\": {\n                \"BannerComponent.js\": {},\n                \"CheckoutComponent.js\": {},\n                \"ProductsListComponent.js\": {},\n                \"NavbarComponent.js\": {},\n                \"CartComponent.js\": {},\n                \"ProductDetailComponent.js\": {}\n            }, \n            \"pages\": {\n                \"CheckoutPage.js\": {},\n                \"HomePage.js\": {},\n                \"Page404.js\": {},\n                \"CartPage.js\": {},\n                \"Product.js\": {}\n            },\n            \"resourses\": {\n                \"images\": {},\n                \"fonts\": {}\n            },\n            \"reducers\": {\n                \"cartReducer.js\": {},\n                \"productsReducer.js\": {}\n            },\n            \"actions\": {\n                \"addToCart.js\": {},\n                \"removeFromCart.js\":\n\n{\n  \"pages\": {\n    \"index.js\": {\n      \"path\": \"/\",\n      \"component\": \"HomePage\"\n    },\n    \"products\": {\n      \"index.js\": {\n        \"path\": \"/products\",\n        \"component\": \"ProductListing\"\n      },\n      \"single-product.js\": {\n        \"path\": \"/products/{id}\",\n        \"component\": \"SingleProduct\"\n      }\n    },\n    \"cart.js\": {\n      \"path\": \"/cart\",\n      \"component\": \"Cart\"\n    },\n    \"checkout.js\": {\n      \"path\": \"/checkout\",\n      \"component\": \"Checkout\"\n    }\n  },\n  \"components\": {\n    \"HomePage.js\": {},\n    \"ProductListing.js\": {},\n    \"SingleProduct.js\": {},\n    \"Cart.js\": {},\n    \"Checkout.js\": {},\n    \"ProductItem.js\": {},\n    \"CartItem.js\": {}\n  },\n  \"routes\": {\n    \"/\": {\n      \"component\": \"HomePage\"\n    },\n    \"/products\": {\n      \"component\": \"ProductListing\"\n    },\n    \"/products/{id}\": {\n      \"component\": \"SingleProduct\"\n    },\n    \"/cart\": {\n      \"component\": \"Cart\"\n    },\n    \"/checkout\": {\n      \"component\": \"Checkout\"\n    }\n  }\n}",
        temperature=1,
        max_tokens=2057,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    print(response)

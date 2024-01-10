import traceback
from flask import Flask, Blueprint, render_template, request, redirect, url_for, session
from model import *
from utils import *

from decimal import Decimal
from langchain.agents import create_csv_agent

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from dotenv import load_dotenv

from langchain import OpenAI, SQLDatabase, SQLDatabaseChain

load_dotenv()

products_blueprint = Blueprint("products", __name__)


# Products Page
@products_blueprint.route("/products", methods=["GET"])
def products_page():
    # Check if the user is logged in by looking for "username" in the session data.
    if "username" in session:
        # Get the username from the session data.
        username = session["username"]

        # Establish a connection to the database and create a cursor.
        cur = mysql.connection.cursor()

        # SQL query to fetch product data from the database for the given storename.
        query = "SELECT p.productid, p.ProductName, p.Productdesc, p.sellingprice, p.discountedprice, p.category, p.quantitysold, p.productlikes, p.productratings, p.productratingsamt, p.shippingtype, p.shipfrom, w.watchlistId FROM product p INNER JOIN store s ON p.storeid = s.storeid LEFT JOIN watchlist w ON p.productid = w.watched_id WHERE s.storename = %s;"

        # Execute the SQL query with the provided storename and fetch the data.
        cur.execute(query, (username,))
        fetchdata = cur.fetchall()

        # Strip any leading or trailing whitespace from the fetched data and create a new list.
        stripped_data = [
            [str(item).strip() if item is not None else None for item in row]
            for row in fetchdata
        ]

        # Close the database cursor.
        cur.close()

        # Render the "products.html" template with the fetched and stripped data and the username.
        return render_template(
            "products/products.html", data=stripped_data, username=username
        )
    else:
        # If the user is not logged in, return a message indicating that the user is not logged in.
        return "User not logged in"



''' 
                                                    view_store Function - Edit Product Page
    Description: 
        The view_store function handles the "editProduct" page for a specific product identified by the given product_id. 
        It retrieves comprehensive product information from the database based on the provided product_id, including details such as product name, description, selling price, discounted price, category, quantity sold, product likes, ratings, and shipping details. 
        The function ensures the correct product's data is fetched and displayed on the "editProduct" page, allowing users to make necessary updates and modifications to specific fields as needed.
'''
@products_blueprint.route("/editProduct/<int:product_id>", methods=["GET"])
def view_store(product_id):
    # Establish a connection to the database and create a cursor.
    cur = mysql.connection.cursor()

    # SQL query to fetch product data from the database for the given product_id.
    cur.execute(
        "SELECT p.productid,p.ProductName,p.Productdesc,p.sellingprice,p.discountedprice,p.category,p.quantitysold,p.productlikes,p.productratings,p.productratingsamt,p.shippingtype,p.shipfrom FROM product p INNER JOIN store s ON p.storeid = s.storeid WHERE p.productid = %s;",
        (product_id,),
    )
    fetchdata = cur.fetchall()

    # Close the database cursor.
    cur.close()

    # Render the "editProduct.html" template with the fetched data for the specified product_id.
    return render_template("products/editProduct.html", data=fetchdata)


'''
                                                    optimize Function - Optimize Product Page
    Description:
        The optimize function handles the "optimizeProduct" page for a specific product identified by the given product_id. It retrieves comprehensive product information from the database, including details such as product name, description, selling price, discounted price, category, quantity sold, product likes, ratings, and shipping details. 
        The function also calculates pricing statistics for products in the same category and with ratings between 1 and 5.
        Additionally, the function determines the average rating for each stripped category, which is derived from the product's full category by removing the first three values. 
        It presents common keywords from the top-rated products in the same category and with ratings between 1 and 5.

'''
@products_blueprint.route("/optimizeProduct/<int:product_id>", methods=["GET"])
def optimize(product_id):
    # Get a cursor object to execute SQL queries
    cur = mysql.connection.cursor()

    # Retrieve product data from the 'product' table and store it in 'fetchdata'
    cur.execute(
        "SELECT p.productid,p.ProductName,p.Productdesc,p.sellingprice,p.discountedprice,p.category,p.quantitysold,p.productlikes,p.productratings,p.productratingsamt,p.shippingtype,p.shipfrom FROM product p INNER JOIN store s ON p.storeid = s.storeid WHERE p.productid = %s;",
        (product_id,),
    )
    fetchdata = cur.fetchall()

    # Retrieve the category data of the product
    cur.execute(
        "SELECT p.category FROM product p INNER JOIN store s ON p.storeid = s.storeid WHERE p.productid = %s;",
        (product_id,),
    )
    category_data = cur.fetchall()

    # Split the categories and remove the first 3 values to get 'stripped_categories'
    categories = category_data[0][0].split(";")
    stripped_categories = [category.strip() for category in categories][3:]

    # Retrieve pricing data for products in the same category and with ratings between 1 and 5
    if stripped_categories:
        cur.execute(
            """
            SELECT p.sellingprice, p.discountedprice, p.productid
            FROM product p
            INNER JOIN (
                SELECT productid
                FROM product
                WHERE category LIKE %s
                AND productratings BETWEEN 1 AND 5
                ORDER BY productratings DESC
                LIMIT 10
            ) AS subquery ON p.productid = subquery.productid
            WHERE p.category LIKE %s;
            """,
            ["%" + stripped_categories[0] + "%", "%" + stripped_categories[0] + "%"],
        )

    pricing = cur.fetchall()
    # Retrieve the avg price, max price, min price for products in the same category and with ratings between 1 and 5
    cur.execute(
        """
        SELECT 
            ROUND(AVG(p.sellingprice), 2) AS avg_selling_price,
            ROUND(AVG(p.discountedprice), 2) AS avg_discounted_price,
            ROUND(MAX(p.sellingprice), 2) AS max_selling_price,
            ROUND(MIN(p.discountedprice), 2) AS min_discounted_price
        FROM product p
        INNER JOIN (
            SELECT productid
            FROM product
            WHERE category LIKE %s
            AND productratings BETWEEN 1 AND 5
            ORDER BY productratings DESC
            LIMIT 10
        ) AS subquery ON p.productid = subquery.productid
        WHERE p.category LIKE %s;
    """,
        ["%" + stripped_categories[0] + "%", "%" + stripped_categories[0] + "%"],
    )
    avgprice = cur.fetchall()

    # Calculate the average rating for each stripped category and get all ratings
    if stripped_categories:
        average_ratings = []
        all_ratings = []
        for category in stripped_categories:
            cur.execute(
                """
                SELECT p.productratings
                FROM product p
                INNER JOIN store s ON p.storeid = s.storeid
                WHERE p.category LIKE %s
                AND p.productratings BETWEEN 1 AND 5
                ORDER BY p.productratings DESC
                """,
                ["%" + category + "%"],
            )
            ratings = cur.fetchall()
            all_ratings.extend([rating[0] for rating in ratings])
            average_rating = (
                sum([rating[0] for rating in ratings]) / len(ratings)
                if len(ratings) > 0
                else 0
            )
            average_ratings.append(average_rating)

    # Retrieve top-rated products and their product names in the same category and with ratings between 1 and 5
    cur.execute(
        """
        SELECT p.ProductName
        FROM product p
        INNER JOIN (
            SELECT productid
            FROM product
            WHERE category LIKE %s
            AND productratings BETWEEN 1 AND 5
            ORDER BY productratings DESC
            LIMIT 10
        ) AS subquery ON p.productid = subquery.productid
        WHERE p.category LIKE %s;
        """,
        ["%" + stripped_categories[0] + "%", "%" + stripped_categories[0] + "%"],
    )
    top_rated_products = cur.fetchall()

    cur.close()

    # Tokenize product names and create a list of product keywords
    product_keywords = []
    for product in top_rated_products:
        product_name = product[0]
        tokens = word_tokenize(product_name)  # Tokenization
        product_keywords.append(tokens)

    # Flatten the list of product keywords
    flattened_keywords = [
        keyword for sublist in product_keywords for keyword in sublist
    ]

    # Count the frequency of each keyword
    keyword_counts = Counter(flattened_keywords)

    # Set the frequency threshold for common keywords
    frequency_threshold = 2

    # Filter out keywords that don't meet the frequency threshold and exclude symbols/numbers
    filtered_common_keywords = [
        (keyword, count)
        for keyword, count in keyword_counts.items()
        if count >= frequency_threshold and re.match(r"^[a-zA-Z]+$", keyword)
    ]

    # Render the 'optimizeProduct.html' template with the retrieved data and calculated values
    return render_template(
        "products/optimizeProduct.html",
        data=fetchdata,
        price=pricing,
        avgprice=avgprice,
        keywords=filtered_common_keywords,
        ratingData=all_ratings,
        avgrating=round(average_rating, 2),
    )

'''
                                delete_product Function - Delete Product
    Description: 
        The delete_product function handles the deletion of a specific product from the database. 
        It takes the product ID from the request form and executes a delete query to remove the corresponding product record from the database.
        If the deletion is successful, it redirects the user to a success page ("/success"). 
        If any errors occur during the deletion process, the function catches the exception and redirects the user to a 404 error page ("/404").
'''
@products_blueprint.route("/delete_product", methods=["POST"])
def delete_product():
    product_id = request.form.get("id")
    print(product_id)
    # Connect to MySQL
    conn = mysql.connection
    cursor = conn.cursor()

    try:
        # Execute the delete query
        query = "DELETE FROM product WHERE productid = %s"
        cursor.execute(query, (product_id,))
        conn.commit()
        return redirect("/success")
    except Exception as error:
        # Handle any errors that occur during the deletion
        print(f"Error deleting product: {error}")
        return redirect("/404")

    finally:
        # Close the cursor
        cursor.close()

'''
                                            add_product Function - Add Product Page
    Description: 
        The add_product function renders the "addProduct.html" page, allowing users to add a new product to the database. 
        It checks if the user is logged in, retrieves the username from the session, and establishes a database connection with a cursor.
        SQL queries are executed to fetch the storeId associated with the current username from the "store" table, distinct product categories from the "product" table, and distinct shipfrom values from the "product" table.
        The fetched data is cleaned, removing unnecessary characters.
        The function then closes the database cursor and renders the "addProduct.html" template with the fetched data, allowing users to add a new product with product details, category selection, and shipping location
'''
@products_blueprint.route("/addProduct", methods=["GET"])
def add_product():
    # Check if the user is logged in by verifying if their username is present in the session.
    if "username" in session:
        # Get the username from the session.
        username = session["username"]

        # Establish a connection to the database and create a cursor.
        cur = mysql.connection.cursor()

        # SQL query to get the storeId from the "store" table for the current username.
        query = "SELECT storeId from store WHERE storename = %s;"
        # SQL query to retrieve distinct product categories from the "product" table.
        category_query = "SELECT distinct category from product;"
        # SQL query to retrieve distinct shipfrom values from the "product" table.
        shipfrom_query = "SELECT distinct shipFrom from product;"

        # Execute the query to get the storeId for the current username.
        cur.execute(query, (username,))
        fetchdata = cur.fetchall()

        # Execute the category_query to get distinct product categories and remove unnecessary characters.
        cur.execute(category_query)
        category_data = cur.fetchall()
        category = [re.sub(r"[\(\),]", "", item[0]) for item in category_data]

        # Execute the shipfrom_query to get distinct shipfrom values and remove unnecessary characters.
        cur.execute(shipfrom_query)
        shipfrom_data = cur.fetchall()
        shipfrom = [re.sub(r"[\(\),]", "", item[0]) for item in shipfrom_data]

        # Close the database cursor.
        cur.close()

        # Render the "addProduct.html" template with the fetched data and categories.
        return render_template(
            "products/addProduct.html",
            data=fetchdata,
            category=category,
            shipfrom=shipfrom,
        )

'''
                                        add_productDB Function - Add Product to Database
    Description: 
        The add_productDB function is responsible for processing the form data submitted by users on the "addProduct" page and adding a new product to the database.
        It extracts the relevant product information from the form and creates a new product object with all the required attributes.
        The function then inserts the new product into the "product" table in the database.
        After successfully committing the changes to the database, the function redirects the user to a success page or back to the "addProduct" page.
'''
@products_blueprint.route("/addProductDB", methods=["POST"])
def add_productDB():
    # Extract data from the form
    product_name = request.form["ProductName"]
    product_description = request.form["ProductDescription"]
    product_categories = request.form["ProductCategories"]
    selling_price = request.form["sellingPrice"]
    storeID = request.form["id"]
    discount_percentage = request.form["discountPercentage"]
    discounted_price = (float(selling_price) * (100 - float(discount_percentage))) / 100
    quantity = request.form["Quantity"]
    free_shipping = request.form.get(
        "freeShipping"
    )  # It will be 'Free shipping' if checked or None if not checked
    shipFrom = request.form["shipFrom"]
    # Extract hidden fields
    product_slug = request.form["productSlug"]
    product_likes = request.form["productLikes"]
    product_rating = request.form["productrating"]
    product_ratings_amt = request.form["productratingsamt"]

    # Create a new Product object with the extracted data
    product = {
        "ProductName": product_name,
        "Productdesc": product_description,
        "sellingprice": selling_price,
        "discountedprice": discounted_price,
        "category": product_categories,
        "quantitysold": quantity,
        "productlikes": product_likes,
        "productratings": product_rating,
        "productratingsamt": product_ratings_amt,
        "shippingtype": "Free shipping" if free_shipping else "",
        "shipfrom": shipFrom,
        "productSlug": product_slug,
        "StoreId": storeID,
    }

    # Execute the SQL query to insert the new product into the database
    cur = mysql.connection.cursor()
    cur.execute(
        """
        INSERT INTO product (
            ProductName, Productdesc, sellingprice, discountedprice, category, quantitysold, 
            productlikes, productratings, productratingsamt, shippingtype, shipfrom, 
            productSlug, storeid
        )
        VALUES (
            %(ProductName)s, %(Productdesc)s, %(sellingprice)s, %(discountedprice)s, %(category)s, 
            %(quantitysold)s, %(productlikes)s, %(productratings)s, %(productratingsamt)s, 
            %(shippingtype)s, %(shipfrom)s, %(productSlug)s, %(StoreId)s
        )
    """,
        product,
    )

    # Commit the changes to the database
    mysql.connection.commit()

    # Close the cursor
    cur.close()

    # Redirect the user to a success page or back to the add product page
    # You can customize this URL to match your application's structure
    return redirect(
        url_for("success")
    )  # Replace 'products.add_product_page' with the actual URL of the add product page



'''
                    generate_listing Function - Generate Product Listing
    Description: 
        This function generates a product listing using an AI language model.
        It interacts with the database and an AI language model to generate relevant product information.
        The user provides product categories and a product description through a form.
        The function prepares the data and prompt for the AI language model to generate a response.
        The AI model suggests three product names, provides keywords for the product name, and suggests a product description.
        The function returns the AI-generated product listing as the response.
'''
@products_blueprint.route("/generate_listing", methods=["POST"])
def generate_listing():
    # Initialize a SQLDatabase object with the database URI.
    db = SQLDatabase.from_uri("mysql+pymysql://root:root@localhost/dbms")

    # Initialize an OpenAI object with the OpenAI API key and other settings.
    llm = OpenAI(
        openai_api_key="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",
        temperature=0,
        verbose=True,
    )

    # Set up the SQLDatabaseChain with the OpenAI model and the database.
    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

    # Extract data from the form submitted by the user.
    product_categories = request.form["ProductCategories"]
    product = request.form["aiInput"]

    # Prepare the initial data and prompt to be passed to the LLM (Language Model) for generating a response.
    select_table = "this query you only need to understand product data"
    bot = "you are a product name generator bot, given the data, description of a product, and category of product"
    data = "the data is data of current listed products. It has a total of 14 columns but you only need productName (name of product), productDesc (product description), selling price (selling price of product), discounted price (price of product after discount), category (category of product concatenated based on their subcategory), products sold (number of products sold), product likes (number of users that like this product), product rating (overall rating of a product by past buyers, max is 5/5), and productRatingsAmt (which is the number of users that rate this product)"
    analysis = "what constitutes a good product is one that has a high rating, high quantity sold, and high product ratings amount"

    # Prepare the user input for the LLM prompt.
    user_input = (
        "based on the",
        product,
        "in",
        product_categories,
        " generate a good product name so that",
        product,
        " will be a good product. Give me 3 suggestions of names and a short write-up on keywords I should include in my product name. Also, give me a suggested description.",
    )

    # Prepare the designed output for the LLM prompt.
    designed_output = "your reply should be strictly in this format. Based on the product name you provided, here are the 3 names I suggest:\n\n<name suggestions>\n\nThese are the keywords you should include:\n\n<keywords>\n\nThis is a suggested description:\n\n<suggested description>"

    # Combine the input and output components to form the input for the LLM.
    output = select_table, bot, data, analysis, user_input, designed_output

    # Run the LLM with the prepared input and get the generated response.
    ai_output = db_chain.run(output)

    # Print the LLM's output to the console for debugging purposes.
    print("output of agent:", ai_output)

    # Return the LLM's output as the response to the user's request.
    return ai_output



'''
                         update_product Function - Update Product Details
    Description: 
        This function handles the update of product details based on user input from a form.
        It extracts the form data and calculates the discounted price based on the discount percentage.
        The function performs an SQL update operation to modify the product details in the database.
        After updating the product details, it redirects to a success page or performs any other necessary action.
'''
@products_blueprint.route("/update_product", methods=["POST"])
def update_product():
    # Retrieve the form data
    id = request.form["id"]
    product_name = request.form["ProductName"]
    product_description = request.form["ProductDescription"]
    selling_price = decimal.Decimal(request.form["sellingPrice"])

    # Retrieve and calculate the discounted price based on the discount percentage
    discount_percentage = decimal.Decimal(request.form["discountPercentage"])
    print(discount_percentage)
    discounted_price = (selling_price * (100 - discount_percentage)) / 100

    quantity = request.form["Quantity"]
    free_shipping = request.form.get("freeShipping")  # Checkbox value

    # Perform the update operation using the retrieved data and the ID
    cur = mysql.connection.cursor()
    sql = "UPDATE product SET productName = %s, productDesc = %s, sellingprice = %s, discountedprice = %s, quantitysold = %s, shippingtype = %s WHERE productId = %s"
    params = (
        product_name,
        product_description,
        selling_price,
        discounted_price,
        quantity,
        free_shipping,
        id,
    )

    # Print the SQL statement and parameters for debugging
    print("SQL Statement:", sql)
    print("Parameters:", params)

    # Execute the update query
    cur.execute(sql, params)

    # Commit the changes to the database
    mysql.connection.commit()
    cur.close()

    # Redirect to a success page or perform any other necessary action
    return redirect("/success")

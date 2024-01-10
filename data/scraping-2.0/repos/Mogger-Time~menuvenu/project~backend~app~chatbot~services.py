import os
from http import HTTPStatus

from flask import jsonify
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import ChatOpenAI

from .. import db
from ..menu.models import Items, Ingredients
from ..orders.models import DiningTables, OrderedItems
from ..restaurant.models import Restaurants


class ChatbotService:
    @staticmethod
    def chatbot_query(data):
        query = data['query']

        # converts text file to a format that is processable by the bot
        loader = TextLoader('data.txt')

        # creates an index from the formatted data to generate responses
        index = VectorstoreIndexCreator().from_loaders([loader])

        return jsonify({'status': HTTPStatus.OK, 'message': index.query(query, llm=ChatOpenAI())})

    @staticmethod
    def data_update():
        f = open('data.txt', 'w')

        restaurant = Restaurants.query.first()

        f.write("OUR RESTAURANT\n")
        f.write("Name - " + restaurant.name + "\n")
        f.write("Phone number - " + restaurant.phone + "\n")
        f.write("\n")

        # Lits all items with their ingredients, price and calories
        item_list = Items.query.all()

        for item in item_list:

            f.write(item.name.upper() + "\n")

            #Ingredients
            f.write("Ingredients/Has - ")
            if item.ingredients:
                end = len(item.ingredients)

                index = 1

                for ingredient in item.ingredients:
                    f.write(ingredient.name)
                    if index < end:
                        f.write(", ")
                    index += 1

                f.write("\n")

            else:
                f.write("There is no listed ingredients\n")

            #Price
            f.write("Price - $" + str(item.price) + "\n")

            #Calories
            f.write("Calories - " + str(item.calories) + "\n")

            #Points to redeem
            if item.points_to_redeem:
                f.write("Points to redeem - " + str(item.points_to_redeem) + "\n")
            else:
                f.write("Points to redeem - Not redeemable with points\n")

            #Points earned
            if item.points_earned:
                f.write("Points earned - " + str(item.points_earned) + "\n")
            else:
                f.write("Points earned - No points earnable\n")

            f.write("\n")

        #List all ingredients
        ingredient_list = Ingredients.query.all()

        for ingredient in ingredient_list:
            if ingredient.items:
                f.write("HAS " + ingredient.name.upper() + "\n")
                for item in ingredient.items:
                    f.write(item.name + "\n")

            else:
                f.write("NO ITEMS HAVE"  + ingredient.name.upper() + "\n")

        f.write("\n")

        # Availability of tables
        f.write("BUSY/AVAILABLE\n")
        count = DiningTables.query.filter_by(occupied=False).count()
        f.write("There are currently " + str(count) + " tables available\n")

        f.write("\n")

        # Most popular dish
        f.write("OUR MOST POPULAR DISH\n")
        popular = db.session.query(OrderedItems.item, db.func.count(OrderedItems.item).label('popularity')). \
            group_by(OrderedItems.item).order_by(db.desc('popularity')).first()
        popular_item = Items.query.filter_by(id=popular.item).first()
        f.write(popular_item.name + "\n")

        f.write("\n")

        # Helper lines for chatbot to learn
        f.write("HOW DO I?\n")
        f.write("Fitness - You can connect your fitness app through your profile in the top right of the screen\n")
        f.write("Assistance - You can request assistance with the 'Assist' button below your order list\n")
        f.write("Bill - You can request the bill with the 'Bill' button below your order list\n")

        f.write("\n")

        # Family Guy 3 hour compilation
        f.write("FAMILY GUY\n")
        f.write("Here you go: https://www.youtube.com/watch?v=qrhFlCoNun0\n")

        f.close()

        return jsonify({'status': HTTPStatus.OK, 'message': 'Data updated'})

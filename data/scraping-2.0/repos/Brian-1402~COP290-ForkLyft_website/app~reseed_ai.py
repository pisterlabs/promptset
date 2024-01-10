import openai
import os
from sqlalchemy import text
from forklyft_app.db import get_db
from forklyft_app import create_app
from forklyft_app.forklyft import get_menu

# openai.organization = os.getenv("OPENAI_ORGANIZATION")
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


"""
create table menus ( menu_id integer primary key auto_increment, image_url text not null, restaurant_id integer, food_type text not null, food_name text not null, food_price integer not null, food_desc text );
"""


def update_menu_item(menu_id, new_desc):
    app = create_app()
    with app.app_context():
        with get_db().connect() as conn:
            conn.execute(
                text(
                    "UPDATE menus SET food_desc = :new_desc WHERE (menu_id = :menu_id)"
                ),
                {"new_desc": new_desc, "menu_id": menu_id},
            )


def access_menus():
    menus = get_menu()
    for row in menus:
        menu_id = row[0]
        food_name = row[4]
        food_price = row[5]


prompt = f"""
Generate a list of three made-up book titles along \
with their authors and genres. 
Provide them in JSON format with the following keys: 
book_id, title, author, genre.
"""

print(get_completion(prompt))

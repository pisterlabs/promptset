import os
import openai
import datagenconfig as dgc
import pandas as pd
from werkzeug.security import generate_password_hash
import csv
from random import randrange, random, randint, uniform, sample
from math import floor, ceil
from faker import Faker
from datetime import date
import time

def get_csv_writer(f):
    return csv.writer(f, dialect='unix')

#SETUP
openai.organization = dgc.config['ORG_KEY']
openai.api_key = os.getenv(dgc.config['SECRET_KEY'])
openai.api_key=dgc.config['SECRET_KEY']
model="text-ada-001"
max_tokens=750
temp=0.7


csv_delimiter = "^"


num_df_rows = 2

rating_map = {1: 'negative', 2: 'negative',
    3: 'neutral', 4: 'positive', 5: 'positive'}

num_sales = {}

num_reviews = {}

num_conversations = {}

price_by_id_tuple = {}

num_sellers = {}

orders_by_user = {}

valid_sellers = {}

num_users = 109153
# num_users = 1000

fake = Faker()

def gen_data():
    start = time.perf_counter()

    products_df = pd.read_csv("ai_generated/products.csv", sep=csv_delimiter)

    global num_df_rows 
    # num_df_rows = products_df.shape[0]
    num_df_rows = 1000

    global num_users
    num_users = 5000

    print("Users...")
    users_df = gen_users()

    users_df = remove_corrupted_data(users_df)
    users_df.to_csv('ai_generated/user_data.csv', sep=csv_delimiter, na_rep='')
    print("Generated ", users_df.shape[0], users_df.shape[1])
    print("Sellers...")

    set_num_sales(products_df)
    set_num_reviews(products_df)
    set_num_conversations(products_df)
    set_num_sellers(products_df)


    users = [i for i in range(num_users)]
    sellers_df = gen_sellers(products_df, users)

    sellers_df = remove_corrupted_data(sellers_df)
    sellers_df.to_csv('ai_generated/seller_data.csv', sep=csv_delimiter, na_rep='')
    print("Generated ", sellers_df.shape[0], sellers_df.shape[1])
    print("Sales...", end=' ', flush=True)

    sales_df = gen_sales(products_df)

    sales_df = remove_corrupted_data(sales_df)
    sales_df.to_csv('ai_generated/sale_data.csv', sep=csv_delimiter, na_rep='')
    print("Generated ", sales_df.shape[0], sales_df.shape[1])
    print("Conversations... ")

    conversations_df = gen_conversations(products_df, sellers_df)

    conversations_df = remove_corrupted_data(conversations_df)
    conversations_df.to_csv('ai_generated/conversation_data.csv', sep=csv_delimiter, na_rep='')
    print("Generated ", conversations_df.shape[0], conversations_df.shape[1])
    print("Reviews...")

    reviews_df = gen_reviews(sellers_df, sales_df, products_df)

    reviews_df = remove_corrupted_data(reviews_df)
    reviews_df.to_csv('ai_generated/review_data.csv', sep=csv_delimiter, na_rep='')
    print("Generated", reviews_df.shape[0], reviews_df.shape[1])
    print("Carts...")

    cart_df = gen_carts(sellers_df,users)

    cart_df = remove_corrupted_data(sellers_df)
    cart_df.to_csv('ai_generated/cart_data.csv', sep=csv_delimiter, na_rep='')

    print("Generated", cart_df.shape[0], cart_df.shape[1])
    # users_df = remove_corrupted_data(users_df)
    # users_df.to_csv('ai_generated/Users.csv', sep=csv_delimiter, na_rep='')

    end = time.perf_counter()

    print((end - start)/60)

def remove_corrupted_data(df):
    df_cols = df.columns.values.tolist()
    for col in df_cols:
        if(type(df[col]) == str):
            original = df.shape
            df = df[df[col].str.contains(csv_delimiter) == False]
            lost_delim = original - df.shape
            if lost_delim > 0:
                print("Lost %d rows beacuse of \"%s\" values in column %s", lost_delim, csv_delimiter, col)
    return df

def gen_carts(sellers_df, users):

    carts_df = pd.DataFrame(columns=['uid', 'pid', 'seller_id', 'quantity', 'price'])
    # carts_df = pd.DataFrame(columns=['ASIN', 'cart_seller_id', 'cart_buyer_id','cart_quantity', 'cart_in-cart'])

    for i in range(0,sellers_df.shape[0]):

        asin = int(sellers_df.at[i, 'seller_product_id'])
        num_sold = ceil(num_sales[asin]*random()/2)
        sales = sample(users, num_sold)
        prod_id = int(sellers_df.at[i, 'seller_product_id'])

        for j in range(len(sales)):
            num_sellers = len(valid_sellers[prod_id])
            seller_index = randint(0, num_sellers - 1)
            seller_id = valid_sellers[prod_id][seller_index]
            quant_upper_bound = ceil(num_sold*0.05)
            quantity = randint(1, quant_upper_bound)
            carts_df.loc[len(carts_df.index)] = [j, prod_id, seller_id, quantity, price_by_id_tuple[seller_id, prod_id]] 
    
    return carts_df


def gen_users():
    global num_users
    users_df = pd.DataFrame(columns=["user_id","user_email","user_password","user_firstname","user_lastname","user_address","user_city","user_state","user_balance"])
    for id in range(num_users):
        profile = fake.profile()
        email = profile['mail']
        plain_password = f'pass{id}'
        password = generate_password_hash(plain_password)
        name_components = profile['name'].split(' ')
        firstname = name_components[0]
        lastname = name_components[-1]
        user_address = profile['address']
        user_city = profile['current_location'][0]
        user_state = profile['current_location'][1]
        balance = f'{str(fake.random_int(max=500))}.{fake.random_int(max=99):02}'
        users_df.loc[len(users_df.index)] = [id, email,password, firstname, lastname, user_address, user_city, user_state, balance]
    users_df = users_df.drop_duplicates(subset=['user_id'])
    users_df = users_df.drop_duplicates(subset=['user_email'])
    users_df.reset_index(inplace=True, drop=True)
    users_df['user_id'] = users_df.index
    num_users = users_df.shape[0]
    return users_df

def gen_conversations(df, sellers_df):
    # conversations_df = pd.DataFrame(columns=['ASIN', 'conversation_seller_id', 'conversation_product_id','conversation_recepient_id', 'conversation_sender_id', 'conversation_message'])
    conversations_df = pd.DataFrame(columns=['conversation_seller_id', 'conversation_product_id','conversation_recepient_id', 'conversation_sender_id', 'conversation_message'])

    users = [i for i in range(num_users)]

    for i in range(0, sellers_df.shape[0]):
        pid = int(sellers_df.at[i, 'seller_product_id'])
        seller_id = sellers_df.at[i, 'seller_id']
        product_id = int(sellers_df.at[i, 'seller_product_id'])
        curr_num_conversations = num_conversations[pid]
        inquiring_users = sample(users, curr_num_conversations)
        for j in range(len(inquiring_users)):
            user_id = inquiring_users[j]
            title = df.at[product_id,'Title']
            if seller_id != user_id:
                prompt = "Write a message from a buyer to a seller about \'" + title + "\'"
                message = "placeholder"
                # message = openai.Completion.create(
                #     model=model,
                #     prompt=prompt,
                #     max_tokens=max_tokens,
                #     temperature=temp
                # )
                conversations_df.loc[len(conversations_df.index)] = [seller_id, product_id, seller_id, user_id, message]
                prompt = "Write a message from a seller to a buyer about \'" + title + "\'"
                message = "placeholder"
                # message = openai.Completion.create(
                #     model=model,
                #     prompt=prompt,
                #     max_tokens=max_tokens,
                #     temperature=temp
                # )
                conversations_df.loc[len(conversations_df.index)] = [seller_id, product_id, user_id, seller_id, message]

    return conversations_df


def gen_sellers(df, users):
    sellers_df = pd.DataFrame(columns=['seller_id', 'seller_product_id', 'seller_quantity', 'seller_price'])

    for i in range(0,num_df_rows):
        asin = i
        curr_num_sellers = num_sellers[asin]
        sellers = sample(users, curr_num_sellers)
        orig_price = df.at[i,'Price']
        valid_sellers[i] = sellers
        for j in range(len(sellers)):
            price = float(get_price(orig_price))
            quantity = randint(1,10000)
            price_by_id_tuple[(sellers[j], i)] = price
            sellers_df.loc[len(sellers_df.index)] = [sellers[j], i, quantity, price] 
    return sellers_df

def get_price(price):
    lower_bound = floor(price*0.9)
    upper_bound = ceil(price*1.1) 
    integer_component = randint(lower_bound, upper_bound)
    decimal_component = randint(0,100)
    return f'{str(integer_component)}.{decimal_component:02}'


def gen_sales(df):
    # sales_df = pd.DataFrame(columns=['ASIN', 'sale_seller_id', 'sale_buyer_id','sale_quantity', 'sale_sell_date', 'sale_sell_time', 'sale_fullfilled'])
    sales_df = pd.DataFrame(columns=['buyer_id','order_number', 'product_id', 'seller_id', 'quantity', 'price','sell_date','sell_time', 'fullfill_date'])
    counter = 0

    order_num = 0
    for i in range(0,num_df_rows):

        asin = i
        users = [i for i in range(num_users)]
        num_sold = num_sales[asin]
        sales = sample(users, num_sold)
        for j in range(len(sales)):
            counter += 1
            if counter % 250 == 0:
                print(f'{counter}', end=' ', flush=True)
            num_sellers = len(valid_sellers[i])
            seller_index = randint(0, num_sellers - 1)
            seller_id = valid_sellers[i][seller_index]
            sell_date = fake.date()
            fullfilled = randint(0,1)
            fullfilled = bool(fullfilled)
            fullfilldate = fake.date()
            sell_time = fake.time()
            quant_upper_bound = ceil(num_sold*0.5)
            quantity = randint(1, quant_upper_bound)
            sales_df.loc[len(sales_df.index)] = [j, order_num, i, seller_id, quantity, price_by_id_tuple[(seller_id, i)] ,sell_date,sell_time,fullfilldate] 
            order_num += 1
    return sales_df


def gen_reviews(sellers_df, sales_df, df):
    # reviews_df = pd.DataFrame(columns=['ASIN', 'review_seller_id', 'review_buyer_id', 'prod_id', 'review', 'rating'])
    reviews_df = pd.DataFrame(columns=['uid', 'pid', 'stars', 'review', 'date'])

    for i in range(0,sellers_df.shape[0]):

        asin = int(sellers_df.at[i, 'seller_product_id'])
        title = df.at[asin,"Title"]
        reviews = num_reviews[asin]
        prod_id = sellers_df.at[i, 'seller_product_id']
        seller_id = sellers_df.at[i,'seller_id']

        buyers_df = sales_df[sales_df['seller_id'] == seller_id]
        buyers_list = buyers_df['buyer_id'].tolist()

        reviews = min(len(buyers_list), reviews)
        buyers = sample(buyers_list,reviews)

        for j in range(len(buyers)):
            buyer_id = buyers[j]
            rating = randint(1,5)
            prompt = "Write a " + rating_map[rating] + " review about \'" + title + "\'"
            review = "placeholder"
            # review= openai.Completion.create(
            #     model=model,
            #     prompt=prompt,
            #     max_tokens=max_tokens,
            #     temperature=temp
            # )
            reviews_df.loc[len(reviews_df.index)] = [buyer_id, prod_id, rating, review, fake.date()] 
    
    return reviews_df

def set_num_sales(df):
    for asin in range(df.shape[0]):
        buyers = randint(0,10)
        num_sales[asin] = buyers

def set_num_reviews(df):
    for asin in range(df.shape[0]):
        sales = num_sales[asin]
        reviews = ceil(sales*random())
        if(floor(reviews) < 0 or floor(reviews) >= num_users):
            print(floor(reviews), sales, asin)
        num_reviews[asin] = floor(reviews)

def set_num_conversations(df):
    for asin in range(df.shape[0]):
        sales = num_sales[asin]
        conversations = ceil(sales*random()/2)
        conversations = ceil(conversations)
        num_conversations[asin] = conversations

def set_num_sellers(df):
    for asin in range(df.shape[0]):
        sales = num_sales[asin]
        sellers = randint(1,4)
        sellers = max(ceil(sellers),1)
        num_sellers[asin] = sellers



gen_data()
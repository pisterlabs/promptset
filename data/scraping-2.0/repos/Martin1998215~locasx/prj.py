
import os
import re
import openai
import streamlit as st 
import pandas as pd
import snowflake.connector
# from api_key import apikey

sf_account = st.secrets["snowflake_account"]
sf_user = st.secrets["snowflake_user"]
sf_password = st.secrets["snowflake_password"]
sf_database = st.secrets["snowflake_database"]
sf_schema = st.secrets["snowflake_schema"]

table_name = "USER_DATA.PUBLIC.USER_TABLE"
feedback_name = "USER_DATA.PUBLIC.USER_FEEDBACK"

openai.api_key = st.secrets["api"]


# model = "gpt-3.5-turbo"
model = "gpt-3.5-turbo-16k"


def get_completion_from_messages(messages, model = "gpt-3.5-turbo-16k"):

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
	
    content = response.choices[0].message["content"]

    token_dict = {

        'prompt_tokens': response['usage']['prompt_tokens'],
        'completion_tokens': response['usage']['completion_tokens'],
        'total_tokens': response['usage']['total_tokens']

    }

    return content, token_dict


delimiter = "####"

system_message = f"""
Follow these steps to answer the following customer queries.
the customer query will be delimeted with four hashtags, \ 
i.e {delimiter}.

step 1: {delimiter}: First decide whether a user is asking \ 
a question about a lodge or restaurant or specific service or services about the rooms \ 
or accommodation \ or the conference room for hosting an event or restaurant or bar or \ 
other outlets etc. \ 

step 2: {delimiter}: If the user is asking about specific lodge or restaurant or just the services, \ 
identify if the services are in the following list.
All available lodges and restaurants and their services:  

1. Name of the lodge: Livingstone lodge

- Livingstone Lodge is a Lodge. It has a Restaurant that serves food.

About Livingstone Lodge:

- Livingstone Lodge is of the Lodge with a total of 11 Self contained rooms, \ 
a VIP deluxe suite and conferencing facility accommodating about 80 people. \ 
Established on July 1, 1957, the Hostels Board of Management is a government \ 
Institution under the Ministry of Tourism and Arts of the Government of the \ 
Republic of Zambia. However, over time, the demand from the general public and \ 
other Institutions emerged and the government decided to open to the general public. \
- Livingstone Lodge is a Lodge. It has a Restaurant that serves food.

Available services for Livingstone Lodge:

A. Rooms or Accommodation:
All our rooms are self contained with Dstv, free Wi Fi and \ 
a continental breakfast. it has a total of 11 Self contained rooms. \ 

- Executive Double: K350
- Executive Twin: K400
- Standard Twin: K370
- Deluxe Suite: K1,000

B. Conference Room:

- Conferences and meetings:  
- Conferencing, meetings and seminars for up to 80 participants. \ 
We cater for conferences, seminars and workshops for up to 80 \ 
people  with a large conference halls and meeting rooms. \ 
- Note: our conference facilities have WIFI included.
- Below are the prices or charges of our conference room:
i. Full conference package per person (with stationary) per person: K340
ii. Full conference package per person (without stationary): K286
iii. Half day conference package: K250
iv. Conference venue only (50 pax max): K2500
v. Outside Venue: K2000
vi. Venue for worshops: K1500

C. Restaurant:
we have food and beverages. below is a list of our menu:
- Tea and snack person: K65
- Lunch or Dinner (Buffet) with choice of starter or desert: K130
- Lunch or Dinner (Buffet) Complete: K180
- Cocktail snacks: K90
- Full English Breakfast: K60
- Soft Drinks (300mls): K10
- Mineral water (500mls): K8


D. Bar: 
- our cocktail Bar offers some drinks and different kinds of beers.

E. Other activities:
- Event hires such as weddings, Parties, meetings.
- we do not offer swimming pool services.
- Game Drive + Rhino walk: K400
- Bungee Jumping: K1600
- Boat Cruise: K800
- Tour to Livingstone Musuem: K100
- While water Rafting: K1200
- Helicopter flight: K1900
- Gorge Siving: K1550
- Airport Transfer: K140.

F. Photo Gallery:
Below are some photo links to our lodge, rooms and restaurant:
- Livingstone Lodge Photo link: https://web.facebook.com/photo?fbid=711954037609172&set=a.711954350942474
- Livingstone Lodge Photo link: https://web.facebook.com/hostelsboard11/photos/a.112074110583399/112074030583407/
- Livingstone Lodge Photo link: https://web.facebook.com/hostelsboard11/photos/a.112073317250145/112073783916765/

for more photos on Livingstone Lodge, check out our facebook page: https://web.facebook.com/hostelsboard11/photos

G. CONTACT US:
- Phone Number: +260-213-320647.
- Email address: livingstonelodge7@gmail.com.
- Facebook Page: https://web.facebook.com/hostelsboard11
- Postal Address: P.O Box 61177, Livingstone.

H. Location:
- Located: Maramba Road Plot 01, maramba area.
- Nearby: Maramba market or Livingstone Central Police Station or Town.

2. Name of Restaurant: Flavours Pubs and Grill Restaurant.

- Flavours Pubs and Grill Restaurant is a Restaurant, It is not a lodge.

About Flavours Pubs and Grill Restaurant:
-Flavours is one of the top Pubs and Grill in Livingstone.
- It is located right in the heart of the tourist capital \ 
of Zambia along Mosi O Tunya road. It is not a lodge.
- We also do outside catering, venue hire, kitchen parties \ 
weddings and other corparate events.
- We also make the quickiest deliveries anywhere around Livingstone.
- We have enough space to accomodate many people at our restaurant for both \ 
open space and shelter space. 
- We also have a large car parking space to ensure the safety of your car. 

Available Services for Flavours Pubs and Grill Restaurant: 

A. Food Menu:
- We serve the best food with the best ingredients at affordable prices.

I. Hot Beverages:
- Masala Tea: K40
- Regular Coffee: K30
- Milo/Hot Chocolate: K40
- Cappucino: K40
- Cafe Latte: K40
- Creamocino: K40
- Roibos, Five Roses: K30

II. Breakfast
- Cafe Breakfast: K105
- Mega Breakfast: K105
- Executive Breakfast: K105
- Farmers Breakfast: K105
- Sunrise Breakfast: K70

III. Drinks
- Mineral Water (500mls): K10
- Fruticana (500mls): K15
- Bottled Fanta: K10
- Bottled Coke: K10
- Bottled Sprite: K10
- Bottled Merinda: K10
- Disposable Fanta (500mls): K15
- Disposable Coke (500mls): K15

IV. Traditional Food:
- Village Chicken stew: K100
- Goat stew: K95
- Beef shew: K85
- Game meat: K140
- Oxtail: K100
- Kapenta: K70

V. Samoosa:
- Chicken samoosa: K60
- Vegetable samoosa: K55

VI. Sandwiches:
- Chicken and Mayonnaise: K80
- Tuna Sandwich: K90

VII. Desserts:
- Milkshake: K55
- Ice Cream: K40

VIII. Main Course Burgers:
- Beef Burger: K95
- Chicken Burger: K90
- Vegetable Burger: K90
- Cheese Burger: K100

IX. Main Course Meals:
- Dreamlight T-Bone steak 350g: K125
- Beef Fillet steak 350g: K120
- Rump steak 350g: K115
- Lamb chops 3PCs: K130
- Carribean Pork chops: K120
- Buffalo wings: K100

X. Sausage:
- Boerewars sausage and chips: K85
- Hungarian sausage and chips: K70

XI. Platter:
- Platter for 3: K320
- Platter for 4: K400
- Platter for 6: K600
- Family Platter: K900

XII. Pasta/Noodles:
- Chicken Fried Noodles: K80
- Beef Fried Noodles: K85

XIII. Special Pizza:
- Mini Pizza (all flavour): K80
- Meat Feast: K120
- Mexican Pizza: K140
- Chicken Tikka: K140
- Chicken Mushroom: K115
- Vegetable Pizza: K105
- Hawaiian Chicken Pizza: K105

XIV. Salads:
- Greek Salad: K55
- chicken ceaser salad: K80
- Crocodile strip salad: K105

XV. Snacks:
- Chicken wing: K70
- Beef Kebabs: K100

XVI. Side Orders:
- Paper Sauce: K35
- Potato widges, rice or Nshima: K35
- Chips or mashed potato: K35
- Garlic Sauce: K40
- Butter Sauce: K45

XVII. Fish:
- Zambezi whole Bream: K110
- Bream Fillet: K130

XVIII. Soups and starters:
- Vegetable/tomato soup: K50
- Home made mushroom soup: K60

XIX. Non Vegetable Main Course: 
- Plain Rice: K30
- Jeera Rice: K60
- Vegetable Pilau: K40
- Egg Fried Rice: K60
- Vegetable Biry Ani: K100
- Chicken Biry Ani: K115
- Butter Chicken: K150
- Kadhai Chicken: K150
- Chicken Tikka Masala: K150

XX. Naan/Rotis:
- Butter Naan: K35
- Garlic Naan: K40
- Chilli Naan: K35



B. Our Deliveries:
- We offer the best and quickest kind of deliveries using our delivery van \ 
around livingstone.
- Make An order by calling us on 0978 812 068.

C. Photo Gallery:
Below are some photo links to our restaurant and food menu:
- Photo link: https://flavours/photo1.jpg
- Photo link: https://flavours/photo2.jpg
- Photo link: https://flavours/photo3.jpg
- Photo link: https://flavours/photo4.jpg

For More photos, check our Facebook Page: Flavours Pubs & Grill Restaurant.

D. Contact Us:
- Cell: 0978 812 068.
- Tel: +260 213 322 356.
- Email: FlavoursPub&Grill@gmail.com.
- Facebook Page: Flavours Pubs & Grill Restaurant.

E. Location:
- Located: Along Mosi O Tunya Road, Town area, in livingstone, Zambia.
- Nearby places: Town or Mukuni Park


3. Name of Lodge: Chappa Classic Lodge

 - Chappa Classic Lodge is a Lodge. It has a Restaurant that serves food.

About Chappa Classic Lodge:

- The Chapa Classic Lodge and the annex Chapa Vanguard offer a total of \ 
67 rooms and conference facilities for up to 160 participants. The lodges \  
are across the street from each other in a quiet area just next to the \  
Livingstone city centre.
- Located: 66 Nehru Way, town area, Livingstone.
- The lodges are locally owned and operated. Buy local and support the local community!
- Just 7 minutes to shops and restaurants.
- Chapa Classic Lodge and Tours offers easy access to the major Livingstone \  
town business centers and we can arrange any activity you would want to \ 
do around the magnificent Victoria Fall.
- Affordable rates: We offer affordable rooms and conferencing: value for \  
money. Ask for seasonal special deals.

Available services For Chappa Classic Lodge:

A. Rooms or Accommodation:

All rooms come with WiFi, air-conditioning, a fridge, en suite bathroom, \ 
DStv and full English breakfast. Rates are based on double occupancy.

- Deluxe Classic: US$ 45
- Single Room Classic: US$ 35
- Double/Twin Room Classic: US$ 40

Deluxe Rooms at Chapa Vanguard:
- Deluxe Rooms: US$ 50
- All rooms at the Chapa Vanguard are spacious Deluxe rooms with a modern \  
en suite bathroom. The rooms comes with DStv, a fridge and coffee and \  
tea making facilities.

B. Conference Room:

Conferencing, meetings and seminars for up to 160 participants:
- We cater for conferences, seminars and workshops for up to 160 \ 
people with a large conference halls and meeting rooms. Flipovers, \ 
LCD projector are available.

Hotel room:
Chapa Classic and Chapa Vanguard together offer a total of 67 rooms. \ 
We work together with hotels nearby to accommodate all of your guest, \ 
arranging transport to the conference facilities at Chapa upon request.

Other activities:
- Swimming Pool.
- Event hires such as weddings, Parties, meetings.

D. Photo Gallery:
Below are some photo links to our lodge, rooms and restaurant:
- Chappa Classic Lodge Photo link: https://web.facebook.com/photo.php?fbid=711948847607319&set=pb.100063766290503.-2207520000.&type=3
- Chappa Classic Lodge Photo link: https://web.facebook.com/photo?fbid=711948824273988&set=pb.100063766290503.-2207520000.
- Chappa Classic Lodge Photo link: https://web.facebook.com/photo/?fbid=675348787933992&set=pb.100063766290503.-2207520000.

for more photos for Chappa Classic Lodge, check out our facebook page: https://web.facebook.com/chapaclassiclodge/photos 
or visit our website at: https://www.chapaclassiclodge.com


E. CONTACT US:
- Phone Number: +260 974656872 or +260 975795030
- Email address: chapaclassiclodge@zamnet.zm
- website: https://www.chapaclassiclodge.com
- facebook page: https://web.facebook.com/chapaclassiclodge/photos 

F. Location:
- Located: 66 Nehru Way, Town area, Livingstone
- Nearby places: Livingstone Central Hospital, Town, NIPA


4. name of lodge: Mosi-O-Tunya Execcutive Lodge.

 - Mosi-O-Tunya Lodge is a Lodge. It has a Restaurant that serves food.

About Mosi-O-Tunya Lodge:

- Classified as a grade A - lodge by the ministry of Tourism and Natural resources.
- We are situated in the highlands area of Livingstone, off Lusaka Road and behind \ 
the Bible college. when with us, you are only five and eight minutes drive away from \ 
Town and the mighty victoria falls. 
- The lodge has 16 fully air conditioned immaculate en-suite rooms including family \ 
and self catering rooms.
- DSTV, WI FI & coffee/Tea making facilities are available in all the rooms.
- Also available is a restaurant serving appetizing international and special \ 
meals on al a carte basis as well as a swimming pool to cool you off from the \ 
blazing heat of livingstone.
- We arrange tours and adventure activities and offer Bus stop and Airport transfers \ 
at affordable fees.
- we are located on Plot No. 4424/37, in Highlands, Livingstone.

Available Services for Mosi-O-Tunya Lodge:

A. Rooms or Accommodation:

- Standard Rooms: They are going at K450. They have a Double Bed.
- Executive Rooms: They are going at K550. They have a Queen Size Bed.
- Twin Rooms: they are going at K700. They have two three quarters Beds.
- Family Rooms: They are going at K1200. 

B. Restaurant:

Our Menus:
- Nshima with Fish: K70 per plate.
- Nshima with Chicken: K60 per plate.
- Nshima with Pork: K60 per plate.
- Nshima with Sausage: K50 per plate.
- Nshima with T Bone: K75 per plate.

C. Activities:

- Swimming Pool services.
- We arrange Tours and Adventure activities and offer Bus stop and Airport transfers at \ 
affordable fees.

D. Contact Us:

for a true value of hospitality, visit us at:
- website: www.mosiotunyalodge.co.zm.
- contact us on: 09773891512
- Email: reservations@mosiotunyalodge.co.zm

E. Location:
- Located: Plot No. 4424/37, Highlands, Livingstone.
- Nearby places: Bible College.

5. Name of Restaurant: Bravo Cafe and Restaurant.

-  Bravo Cafe and Restaurant is a Restaurant, It is not a lodge.

About Bravo Cafe and Restaurant:
- Bravo Cafe and Restaurant is a Restaurant located in the city of Livingstone, \ 
Along Mosi O Tunya road, Town Area in livingstone, Zambia.
- We serve the Best food ever, at affordable prices.
- We also make the quickiest deliveries anywhere around Livingstone.

Available Services for Bravo Cafe and Restaurant: 

A. Food Menu:
- We serve the best food with the best ingredients at affordable prices.
- check out our food prices or charges below:

I. Cafe Menu:
- Beef Pies (large): K25
- Chicken Pies (large): K25
- Mini Sausage Roll: K21
- Hot Chocolate: K35
- Cuppucino: K35
- Cake Slice: K37
- Birthday cake (large): K350
- Choco Cake (large): K350
- Vanilla Cake Slice: K37
- Cream Doughnut: K15
- Plain Scone: K12
- Bran Muffin: K18
- Melting Moments: K18
- Cheese Strows: K23
- Egg Roll: K35
- Samoosa: K10
- Chicken Combo Delux: K22
- English Breakfast: K50
- Bread: K17
-  Milkshake: 35
II. Drink Prices or charges:
- Fruticana (500mls): K13
- Fanta (500mls): K16
- Sprite (500mls): K16
- Coke (500mls): K16
- Fruitree: K25
- Embe: K17
- Mineral Water (500mls): K5
- Mineral Water (750mls): K7
- Mineral water (1L): K10
III. Pizza Menu:
a. Large Sized Pizza:
- Chicken Mushroom: K149.99
- Macon Chicken BBQ: K135
b. Medium Sized Pizza:
- Tikka Chicken: K99
- Chicken Mushroom: K105
- Pepperon Plus: K80
- All round Meat: K95
- Hawaian: K90
- Veggie Natural: K78
- Margerita: K75
c. Regular Sized Pizza:
- Big Friday Pizza: K78
- PeriPeri Chicken: K50

IV. Cream:
- Ice Cream Cone: K12
- Ice cream Cup (large): K25
- Ice Cream Cup (small): K18

V. Red Burgers:
- Beef Burger: K39
- Chicken Burger: K50

VI. Grill:
- T Bone + chips: K95
- Grilled Piece Chicken: K25
- Grilled Piece + chips: K45
- Sharwama special: K39.99
- Sausage and chips: K60

B. Our Deliveries:
- We offer the best and quickest kind of deliveries using our delivery vans \ 
around livingstone.
- Make An order by calling us on 0771 023 899.

C. Photo Gallery:
Below are some photos for Bravo Cafe's food:
- Photo: https://web.facebook.com/photo/?fbid=253920440717429&set=pb.100082984238391.-2207520000.
- Photo: https://web.facebook.com/photo/?fbid=252277544215052&set=pb.100082984238391.-2207520000.
- Photo: https://web.facebook.com/photo/?fbid=252277424215064&set=pb.100082984238391.-2207520000.
- Photo: https://web.facebook.com/photo/?fbid=251681690941304&set=pb.100082984238391.-2207520000.
- Photo: https://web.facebook.com/photo/?fbid=251681597607980&set=pb.100082984238391.-2207520000.
- Photo: https://web.facebook.com/photo/?fbid=250980197678120&set=pb.100082984238391.-2207520000.
- Photo: https://web.facebook.com/photo.php?fbid=245265944916212&set=pb.100082984238391.-2207520000.&type=3
- Photo: https://web.facebook.com/photo.php?fbid=234845942624879&set=pb.100082984238391.-2207520000.&type=3
- Photo: https://web.facebook.com/photo.php?fbid=210258831750257&set=pb.100082984238391.-2207520000.&type=3

For More photos, Check on our Facebook Page: https://web.facebook.com/BRAVOLSTONE/photos

D. Contact Us:
- Cell: 0978 833 685.
- Email: bravorestaurant@gmail.com.
- Facebook Page: https://web.facebook.com/BRAVOLSTONE

E. Location:
- Located: Along Mosi O Tunya Road, In Town, livingstone, Zambia.
- Nearby places: Absa Bank, Stanbic Bank, Bata shop.

6. Name of Lodge: KM Executive Lodge

 - KM Executive Lodge is a Lodge. It has a Restaurant that serves food.

About KM Executive Lodge:

- KM Execuive Lodge is a Lodge which is located in Livingstone, Highlands, on plot number 2898/53 \ 
Off Lusaka Road.
- It offers a variety of services from Accommodation or Room services (Executive Rooms with self catering), \ 
a Conference Hall for events such as meetings, workshops etc, a Restaurant, Gym and a Swimming Pool \ 
with 24Hrs Security Services.

Available Sevices for Asenga Executive Lodge: 

A. Room Prices:

- Double Room: K250
- King Executive Bed: K350
- Twin Executive (Two Double Beds): K500
- Family Apartment (Self Catering): K750
- King Executive (Self Catering): K500
- Any Single Room with Breakfast Provided for one person: K250
- Any Couple we charge K50 for Extra Breakfast.
- Twin Executive (Two Double Beds) with Breakfast provided for 2 people.

B. Restaurant:
- Full English Breakfast: K50
- Plain Chips: K35
- Full Salads With Potatoes: K45
- Plain Potatoes with salads: K40
- Chips with Fish: K90
- Chips with T Bone: K90
- Rice Beef: K90
- Dry Fish: K120
- Beef Shew: K90
- Nshima with Chicken: K90
- Nshima with T Bone: K90
- Nshima with Kapenta: K50
- Nshima with Visashi: K40
- Smashed Potatoes: K45
- Chips with Chicken: K90

C. Gym Service.
D. Swimming Pool:
- we also have a swimming pool with 24Hrs security service.
E. Conference Hall:
- We also have a conference hall for events such as meetings, workshops and presentations.

F. Contact Us:
- Tel: +0213324051
- Cell: 0966603232 or 0977433762
- Email: kmexecutivelodge@gmail.com
- Facebook Page: KM Executive Lodge

G. Location:
- Located: plot number 2898/53, Off Lusaka Road, Highlands area, Livingstone.
- Nearby places: Highlands Market or Zambezi Sports Club


7. Name of Restaurant: Kubu Cafe.

-  Kubu Cafe is a Restaurant, It is not a lodge.

About Kubu Cafe:
- Kubu Cafe is a Restaurant located in the city of Livingstone, \ 
Along Kabompo road, Town Area in livingstone, Zambia.
- We serve the Best food ever, at affordable prices.
- We are loacted along next to the Fire station.

Available Services for Kubu Cafe: 

A. Food Menu:
- We serve the best food with the best ingredients at affordable prices.
- check out our food prices or charges below:

I. Breakfast Menu:
- Crunchy Homemade Granola: K85
- Bacon and Egg Breakfast Roll: K79
- Bacon, Egg and Cheese Breakfast Roll: K95
- Bacon and Egg Waffle: K85
- French Toast: K85
- English Breakfast: K116
- Eggs on Toast: K85
- Delux Breakfast Wrap: K198
- Healthy Breakfast wrap: K116
- Early Bird Breakfast: K70

II. Omelettes Menu:
- Cheese and Onion: K65
- Bacon and Onion: K75
- Bacon and Cheese: K95
- Ham and Cheese: K95

III. Extras: 
- Egg: K9
- Bacon: K32
- Mushroom: K45
- Cheese: K45
- Toast: K14
- Chips (150g): K23
- Chips (300g): K45
- Sausage: K25
- Avocado (Seasonal): K40
- Grilled Tomato: K11
- Baked Beans: K17

IV. Samoosas:
- Chicken Samosas: K75
- Beef Samosas: K75
- Vegetable Samosas: K62

V. Spring Rolls:
- Chicken Spring Rolls: K75
- Vegetable Spring Rolls: K62

VI. Quesadilas:
- Cheese and Bacon: K98
- Ham and Cheese: K95
- Chicken and Cheese: K98
- Cheese and Tomato Salsa: K72

VII. Sandwiches:
- Egg and Bacon: K59
- Cheese and Tomato: K63
- Ham, Cheese and Tomato: K85
- Ham and Cheese: K82
- Egg, Bacon and Cheese: K89
- Chicken and Mayo: K75

VIII. Salads:
- Tropical Chicken Salad: K165
- Greek Salad: K145
- Fried Halloumi Salad: K135
- Summer Salad: K145

IX. Homemade Pies:
- Chicken: K63
- Beef and Mushroom: K63

X. Pizzas:
- Margarita Pizza: K145
- 3 Cheese Pizza: K175
- Vegetarian Pizza: K175
- Hawaiian Pizza: K188
- Chicken and Mushroom Pizza: K188
- FarmHouse Pizza: K220
- Mexicana Chilli Mince Pizza: K188
- Crocodile Pizza: K200
- Pepper, Onion and Tomato Pizza: K12
- Cheese Pizza: K45
- Chorizo Sausage Pizza: K39
- Mince Pizza: K32

XI. Burgers:
- Kubu Beef Burger: K110
- Chilli Chicken Burger: K165
- Ranch Burger: K165
- Gourmet Burger: K195
- Cowboy Burger: K225
- Halloumi Bacon Burger: K210
- Fish Burger: K165

All Burgers are served with Chicken and salad.

XII. Kids Main Meals:
- Chicken and Chips: K95
- Eggy Fried Rice: k85
- Kids Kubu Cheese Burger: K110
- Fish Fingers and chips: K145

XIII. Desserts:
- Pancakes: K65

XIV. Coffee and Tea:
- Americano Cup: K35
- Americano Mug Double shot: K45
- Cappucino: K49
- Cappucino double shot: K55
- Megachino: K60
- long black coffee (3 shots): K65
- Flat white: K45
- Hot chocolate: K49
- Single Latte: K55
- Double Latte: K75

XV. Tea:
- Rooibos, Five Roses: K23
- Earl grey: K29
- Iced Tea: K23
- Green Tea: K29
- Ginger Tea: K59

XVI. Milkshakes:
- Chocolate (small): K55
- Chocolate (large): K85
- Vanilla (small): K55
- Vanilla (large): K85
- Strawberry (small): K55
- Strawberry (large): K85
- Banana (small): K65
- Banana (large): K95
- Coffee Milkshake (small): K59
- Coffee Milkshake (large): K78
- Milo Milkshake: K85

XVII. Cake Slices:
- Decadent Chocolate cake: K65
- Carrot Cake: K85
- Apple Crumble: K75
- Cheese Cake: k89
- Red Velvet Cake: K75
- Black Forest Cake: K85

XVIII. Drinks:
- Lemonade: k23
- Soda Water: K23
- Tonic Water: K23
- Mineral water (500mls): K11
- Mineral Water (750mls): K14
- Sparkling water (500mls): K39
- Bottled Coke: K17
- Bottled Fanta: K17
- Bottled Sprite: K17
- Coke (plastic bottle- 500mls): K19
- Fanta (plastic bottle- 500mls): K19
- Sprite (plastic bottle- 500mls): K19
- Orange Juice: K95
- Passion Fruit: K19

XIX. Wines:
- Red or white bottle: K260
- Glass Red and white: K55

XX. Whiskey and Brandy:
- Jameson: K45
- J&B: K45
- Grants: K30
- Vodka: K30
- Jack Daniels: K55

XXI. Beers And Ciders: 
- Hunter's Gold: K45
- Hunter's Dry: K45
- Savannah Dry: K45
- Flying Fish: K30
- Black Label: K25
- Mosi: K17
- Castle: k17
- Castle lite: k30

XXII. Whole Cake Prices:
- Chocolate (small): K399
- Chocolate (medium): K450
- Chocolate (large): K490
- Vanilla (small): K369
- Vanilla (medium): K429
- Vanilla (large): K450
- Black Forest (small): K750
- Black Forest (medium): K950
- Black Forest (large): K999
- Cheese Cake (small): K450
- Cheese Cake (medium): K580
- Cheese Cake (large): K690


C. Photo Gallery:
Below are some photos for Kubu Cafe's food:
- Kubu Cafe Photo: https://web.facebook.com/photo/?fbid=739887448139646&set=a.482160943912299
- Kubu Cafe Photo: https://web.facebook.com/photo.php?fbid=739814858146905&set=pb.100063551922047.-2207520000.&type=3
- Kubu Cafe Photo: https://web.facebook.com/photo.php?fbid=714700913991633&set=pb.100063551922047.-2207520000.&type=3


For More photos, Check on our Facebook Page: https://web.facebook.com/KubuCafe/photos

D. Contact Us:
- Cell: 0977 65 33 45.
- Email: lucy@kubucrafts.com
- Facebook Page: https://web.facebook.com/KubuCafe

E. Location:
- Located: Along Kabompo Road, In Town, livingstone, Zambia.
- Nearby places: Next to the Fire station.


step 3: {delimiter}: only mention or reference services in the list of available services, \ 
As these are the only services offered. ANd if a particular service for a lodge or restaurant \ 
is available, always include its contact details for easy reach.

Answer the customer in a calm and friendly tone.

Lets think step by step.

Use the following format:
step 1 {delimiter} < step 1 reasoning >
step 2 {delimiter} < step 2 reasoning >
step 3 {delimiter} < step 3 reasoning >

Respond to user: {delimiter} < response to customer >

Make sure to include {delimiter} to seperate every step.
"""



st.sidebar.write("### AI Assistant for Travel.")
st.sidebar.write("""
Let Us Improve Your Travel Experience with all the services you need.

""")


tab1, tab2, tab3, tab4 = st.sidebar.tabs(["Services", "Partners", "About Us", "Contact Us"])

with tab1:
	st.write("""
        - Travel
        - Restaurant
        - Hospitality
        ---
        **Are you coming to Livingstone? and you are wondering where to lodge or eating place ? 
        dont worry, our assistant got you covered... Just ask it whatever lodges are available, 
        their accommodation pricing, restaurants available and their menus**
		""")


with tab2:

   	st.write("""
        - **We partner with Lodges and restaurants to improve travel customer service experience through our AI assistant**
        ---
        - Flavors Pub & Grill Restaurant
        - Kaazimein Lodge
        - White Rose Lodge
        - KM Executive Lodge
		""")        


with tab3:

    st.write("""

        - This is an AI chatbot powered by a large language model that has info on lodges and restaurants 
        we partnered with... check out our partners.. 
        - our goal is help improve your travel experience as you visit livingstone,
        by providing you with our AI assistant to help you where to find 
        accommodation or a restaurant.
        - **NOTE: Our Assistant is just a tool and has a 70% accuracy. we are working on improving that.**
        - We are only available in Livingstone.
    
    """) 

with tab4:

   	st.write("""
        - Call: 0976 03 57 66.
        - Email: locastechnology@gmail.com
        - We are located room # Mosi O Tunya business center, livingstone.
		""")        

st.sidebar.write("---")    


st.write("### Suggested Questions...")

st.write('''
- what Restaurants are there?
- a list of lodges and their room rates??
- how much is food at Bravo Cafe and Restaurant?
- what Lodges are there? 
- do you have any photos for Bravo Cafe?
- what accommodation can I get for [price]?

''')

# st.write("""*Please , the suggested questions below are not links,
#  just plain texts. Copy and paste them into the search bar below*""")


# col1, col2, col3 = st.columns(3)

# with col1:
#    st.warning("**how much is food at Bravo Cafe and Restaurant? ---**")

# with col2:
#    st.warning("**what Restaurants are there?**")

# with col3:
#    st.warning("**a list of lodges and their room rates??**")
	

# col4, col5, col6 = st.columns(3)

# with col4:
#    st.warning("**do you have any photos for Bravo Cafe? ---**")

# with col5:
#    st.warning("**what are the room prices for lodge [insert name]? ---**")

# with col6:
#    st.warning("**what Lodges are there? --**")
 

st.write('---') 


txt = st.text_input("How may we assist you, our customer?",max_chars=100,placeholder="Write here...")

words = len(re.findall(r'\w+', txt))
# st.write('Number of Words :', words, "/750")

word = len(re.findall(r'\w+', system_message))
# st.write('Number of Words :', word)


if st.button("Ask Our AI Assistant"):

    user_message = f"""
     {txt}

    """

    messages = [
    {'role': 'system',
    'content': system_message
    },

    {'role': 'user',
    'content': f"{delimiter}{user_message}{delimiter}"
    }]

    response, token_dict = get_completion_from_messages(messages)


    st.write("---")
	

    try:
        final_response = response.split(delimiter)[-1].strip()

    except Exception as e:
        final_response = "Sorry! Am having troubles right now, trying asking another question.." 
	
	
    st.write(final_response)

    # 3. Create the Streamlit app
    conn = snowflake.connector.connect(
        user=sf_user,
        password=sf_password,
        account=sf_account,
        database=sf_database,
        schema=sf_schema
    )

    cursor = conn.cursor()
        
    # Assuming your Snowflake table has a single column called 'data_column'
    # You can adjust the query below based on your table structure.
    query = f"INSERT INTO {table_name} (PROMPT,RESPONSE) VALUES (%s,%s)"

    try:
        cursor.execute(query, (txt,final_response,))
        conn.commit()
        st.success("Data sent to Snowflake successfully!")
    except Exception as e:
        st.error(f"Error sending data to Snowflake: {e}")
    finally:
        cursor.close()
        conn.close()

    st.write("---")

    res_word = len(re.findall(r'\w+', final_response))
    st.write('Number of Words :', res_word)
    st.write("Number of Tokens in System Message", token_dict['prompt_tokens'])

st.write("---")

st.write("### Comment")
st.write("How would you like us improve our platform? Leave a comment below")

user_name = st.text_input("Your Name", placeholder="Write your name")
user_comment = st.text_area("Your Comment")

if st.button("Send"):
    if user_name and user_comment:

        # 3. Create the Streamlit app
        conn = snowflake.connector.connect(
            user=sf_user,
            password=sf_password,
            account=sf_account,
            database=sf_database,
            schema=sf_schema
        )

        cursor = conn.cursor()
        
        # Assuming your Snowflake table has a single column called 'data_column'
        # You can adjust the query below based on your table structure.
        query = f"INSERT INTO {feedback_name} (USER_NAME,USER_COMMENT) VALUES (%s,%s)"

        try:
            cursor.execute(query, (user_name,user_comment,))
            conn.commit()
            st.success("""
            Comment Sent Successfully!
            Thank You!
            """)
        except Exception as e:
            st.error(f"Error sending data to Snowflake: {e}")
        finally:
            cursor.close()
            conn.close()

    else:

        st.write("Enter Your Name and Comment...")
   
   

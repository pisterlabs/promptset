from flask import Flask, jsonify, request
from flask_cors import CORS

from firebase_admin import credentials, firestore, db
import firebase_admin

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from datetime import datetime, timedelta
import holidays

import plotly.express as px

import joblib
from xgboost import XGBRegressor


import requests
from bs4 import BeautifulSoup
import pytz
import time

from pandasai import SmartDataframe
import pandas as pd
from pandasai.llm import OpenAI

cred = credentials.Certificate("./permissions.json")

firebase_admin.initialize_app(cred)

app = Flask(__name__)
cors = CORS(app)

df = pd.read_csv("./food_sales2.csv")
df = df.dropna()

all_dish_id = df['DishID'].unique()

db = firestore.client()

dishes = db.collection("products")

le = LabelEncoder()


def generate_transactions(date, vegetarian_value, price_value, dishID_value):
    return [{'Date': date,
             'DishID': dishID_value,
             'Vegetarian': vegetarian_value,
             'Price': price_value,
             'DayOfWeek': date.strftime('%A'),
             'Occasion': get_occasion_name(date)}]
    
def get_occasion_name(date):
    india_holidays = holidays.India(years=[date.year])
    occasion_name = india_holidays.get(date)
    return str(occasion_name) if occasion_name else 'None'

def generate_dates(start_date, end_date):
    date_range = (end_date - start_date).days
    return [start_date + timedelta(days=i) for i in range(date_range + 1)]

start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 1, 31)
date_range = pd.date_range(start_date, end_date)

all_dates = generate_dates(start_date, end_date)

next_day_date = datetime.strptime('02-01-2023', '%d-%m-%Y').strftime('%Y-%m-%d')


def get_next_id(collection_name):
    # Query the collection to find the maximum ID
    docs = db.collection(collection_name).stream()
    max_id = 0

    for doc in docs:
        current_id = int(doc.id)
        if current_id > max_id:
            max_id = current_id

    return max_id + 1

@app.route('/add-recipe/', methods=['POST'])
def create_product():
    try:
        # Assuming the request body is in JSON format
        req_data = request.get_json()

        # Generate the next ID for the 'products' collection
        next_id = get_next_id('recipies')

        # Add a new document to the 'products' collection with the generated ID
        db.collection('recipies').document(str(next_id)).set({
            'name': req_data['name'],
            'ingredient_list': req_data['ingredient_list'],
            'price_list': req_data['price_list'],
            'quantity_list': req_data['quantity_list'],
            'cost_price': req_data['cost_price'],
            'selling_price': req_data['selling_price'],
            'num_of_dishes': req_data['num_of_dishes'],
            'is_veg': req_data['is_veg']
        })

        return jsonify({'message': 'Product created successfully'}), 200
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500
    
@app.route('/add-collaboration/', methods=['POST'])
def add_collaboration():
    try:
        # Assuming the request body is in JSON format
        req_data = request.get_json()

        # Add a new document to the 'collaborations' collection
        db.collection('collaborations').document().set({
            'restaurantName': req_data['restaurantName'],
            'collaborationDuration': req_data['collaborationDuration'],
            'collaborationDetails': req_data['collaborationDetails'],
            'contactPerson': req_data['contactPerson'],
            'contactEmail': req_data['contactEmail'],
        })

        return jsonify({'message': 'Collaboration details added successfully'}), 200
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500
    
@app.route('/get-collaborations/', methods=['GET'])
def get_collaborations():
    try:
        # Reference to the "collaborations" collection in Firebase
        collaborations_ref = db.collection('collaborations')

        # Fetch all documents from the collection
        collaborations = collaborations_ref.get()

        # Extract data from documents
        data = []
        for doc in collaborations:
            data.append({**doc.to_dict(), 'id': doc.id})

        return jsonify(data), 200
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

    



@app.route('/add-fixed-expense/', methods=['POST'])
def add_fixed_expense():
    try:
        # Assuming the request body is in JSON format
        req_data = request.get_json()

        # Generate the next ID for the 'products' collection
        next_id = get_next_id('restaurants')

        # Add a new document to the 'products' collection with the generated ID
        db.collection('restaurants').document(str(next_id)).set({
            'rent': req_data['rent'],
            'employeeSalaries': req_data['employeeSalaries'],
            'utilities': req_data['utilities'],
            'desiredProfitPercentage': req_data['desiredProfitPercentage'],
            'total_exp': req_data['total_exp'],
            'expected_fluctuation': req_data['expected_fluctuation']
        })

        return jsonify({'message': 'Product created successfully'}), 200
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500
    


@app.route("/dishes/update/<int:dishId>")
def update(dishId):
    
    dish_ref = dishes.document(str(dishId))
    dish = dish_ref.get().to_dict()
    dish["price"] = 100
    dish_ref.set(dish)

    return jsonify({"success": True}), 200

@app.route("/dishes/read/<string:dishId>")
def read(dishId):
    
    dish = dishes.document(dishId).get()
    return jsonify(dish.to_dict()), 200

@app.route("/dishes/create")
def create():
    all_dish_data = []
    
    for dish in dishes.stream():
        dish_data = dish.to_dict()
        all_dish_data.append(dish_data)
        
    last_element_id = all_dish_data[-1]['id']
    
    description = "Created"
    name = "Tandoori"
    price = 300
    
    dishes.document(str(last_element_id + 1)).set({"description": description, "name": name, "id": last_element_id + 1, "price": price})
    return jsonify({"success": True}), 200

@app.route("/dishes/delete/<string:dishId>")
def delete(dishId):
    
    dishes.document(dishId).delete()

    return jsonify({"success": True}), 200



@app.route('/api/read/', methods=['GET'])
def read_products():
    try:
        query = db.collection('recipies')
        response = []

        docs = query.stream()

        for doc in docs:
            selected_item = {
                'id': doc.id,
                'name': doc.to_dict()['name'],
                'ingredient_list': doc.to_dict()['ingredient_list'],
                'price_list': doc.to_dict()['price_list'],
                'quantity_list': doc.to_dict()['quantity_list'],
                'cost_price': doc.to_dict()['cost_price'],
                'selling_price': doc.to_dict()['selling_price'],
                'num_of_dishes': doc.to_dict()['num_of_dishes'],
                'is_veg': doc.to_dict().get('is_veg', None),
            }
            response.append(selected_item)

        return jsonify(response), 200
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500
    
@app.route('/save-selected-data', methods=['POST'])
def save_selected_data():
    try:
        req_data = request.get_json()

        # Assuming your Firebase collection is named 'selectedDishes'
        selected_dishes_ref = db.collection('selectedDishes')

        # Get the current date in DD-MM-YY format
        current_date = datetime.now().strftime('%d-%m-%Y')

        # Loop through the array and add each object to the collection with the current date
        for item in req_data:
            selected_dishes_ref.add({
                'name': item['name'],
                'cost_price': item['cost_price'],
                'selling_price': item['selling_price'],
                'quantity': item['quantity'],
                'id': item['id'],
                'date_added': current_date
            })

        return jsonify({'message': 'Data saved successfully'}), 200

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

# get all the inventory items
@app.route('/get-inventory', methods=['GET'])
def get_inventory():
    try:
        # Assuming your Firebase collection is named 'inventory'
        inventory_ref = db.collection('inventory')

        # Get all documents from the 'inventory' collection
        inventory_data = inventory_ref.stream()

        # Convert data to a list of dictionaries
        inventory_list = []
        for doc in inventory_data:
            item_data = doc.to_dict()
            item_data['id'] = doc.id  # Include the document ID
            inventory_list.append(item_data)

        return jsonify({'inventory': inventory_list}), 200

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500   


 
# send inventory items
@app.route('/save-inventory', methods=['POST'])
def save_inventory():
    try:
        req_data = request.get_json()

        # Assuming your Firebase collection is named 'selectedDishes'
        selected_dishes_ref = db.collection('inventory')

        # Get the current date in DD-MM-YY format
        current_date = datetime.now().strftime('%d-%m-%Y')

        # Loop through the array and add each object to the collection with the current date
        for item in req_data:
            selected_dishes_ref.add({
                'commodity_id': item['commodity_id'],
                'name': item['name'],
                'category': item['category'],
                'unitOfMeasurement': item['unitOfMeasurement'],
                'currentStock': item['currentStock'],
                'minStockThreshold': item['minStockThreshold'],
                'reorderQuantity': item['reorderQuantity'],
                'unitCost': item['unitCost'],
                'date_added': current_date
            })

        return jsonify({'message': 'Data saved successfully'}), 200

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500


@app.route('/get-all-selected-dishes/', methods=['GET'])
def get_all_selected_dishes():
    try:
        # Assuming your Firebase collection is named 'selectedDishes'
        selected_dishes_ref = db.collection('selectedDishes')

        # Retrieve all documents from the 'selectedDishes' collection
        selected_dishes = selected_dishes_ref.stream()

        # Convert Firestore documents to a list of dictionaries
        selected_dishes_list = []
        holiday_calendar = holidays.CountryHoliday('IND')
        for doc in selected_dishes:
            date_weekday = doc.to_dict()['date_added']
            date_object = datetime.strptime(date_weekday, "%d-%m-%Y")
            day_of_week = date_object.strftime("%A")
            holiday_calendar = holidays.CountryHoliday('IND')
            if date_object in holiday_calendar:
                occasion = holiday_calendar.get(date_object)
            else:
                occasion = "None"
            selected_dishes_list.append({
                'DishID': int(doc.to_dict()['id']) + 1,
                'Price': doc.to_dict()['selling_price'],
                'QuantitySold': doc.to_dict()['quantity'],
                'Date': doc.to_dict()['date_added'],
                'Vegetarian': doc.to_dict()['isveg'],
                'DayOfWeek': day_of_week,
                'Occasion': occasion,
            })
            
            df_append_foods = pd.DataFrame(selected_dishes_list)
        
        result_df = pd.concat([df, df_append_foods], ignore_index=True)
        print(result_df.tail())
        
        result_df.to_csv('food_sales2.csv', index=False)

        return jsonify({'selected_dishes': selected_dishes_list}), 200

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500
    
@app.route('/get-all-dishes-openai', methods=['GET'])
def get_all_dishes_openai():
    try:
        # Assuming your Firebase collection is named 'selectedDishes'
        selected_dishes_ref = db.collection('selectedDishes')

        # Retrieve all documents from the 'selectedDishes' collection
        selected_dishes = selected_dishes_ref.stream()

        # Convert Firestore documents to a list of dictionaries
        selected_dishes_list = []
        holiday_calendar = holidays.CountryHoliday('IND')
        for doc in selected_dishes:
            date_weekday = doc.to_dict()['date_added']
            date_object = datetime.strptime(date_weekday, "%d-%m-%Y")
            day_of_week = date_object.strftime("%A")
            holiday_calendar = holidays.CountryHoliday('IND')
            if date_object in holiday_calendar:
                occasion = holiday_calendar.get(date_object)
            else:
                occasion = "None"
            selected_dishes_list.append({
                'dish_id': int(doc.to_dict()['id']) + 1,
                'dish_name': doc.to_dict()['name'],
                'sellingPrice': doc.to_dict()['selling_price'],
                'quantity': doc.to_dict()['quantity'],
                'order_date': doc.to_dict()['date_added'],
                'costPrice': doc.to_dict()['cost_price'],
                'DayOfWeek': day_of_week,
                'Occasion': occasion,
            })
            
            df_append_foods = pd.DataFrame(selected_dishes_list)
        
        df_append_foods.to_csv('./past_month_data.csv', index=False)

        return jsonify({'selected_dishes': selected_dishes_list}), 200

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

@app.route('/read-fixed-exp/', methods=['GET'])
def read_fixed_exp():
    try:
        query = db.collection('restaurants')

        docs = query.stream()
        val = 0
        for doc in docs:
            if(int(doc.id) > int(val)):
                val = doc.id
                selected_item = {
                    'rent': doc.to_dict().get('rent', None),
                    'employeeSalaries': doc.to_dict().get('employeeSalaries', None),
                    'utilities': doc.to_dict().get('utilities', None),
                    'desiredProfitPercentage': doc.to_dict().get('desiredProfitPercentage', None),
                    'total_exp': doc.to_dict().get('total_exp', None),
                    'expected_fluctuation': doc.to_dict().get('expected_fluctuation', None)
                }
                response = selected_item

        return response, 200
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

@app.route("/dishes/alldishes")
def all_dish():
    all_dish_data = []
    recipies = db.collection("recipies")
    for doc_snapshot in recipies.stream():
        doc_data = doc_snapshot.to_dict()
        all_dish_data.append(doc_data)
    
    return jsonify({"documents": all_dish_data}), 200

@app.route("/dishes/topdish", methods=['GET', 'POST'])
def top_dish():
    df = pd.read_csv("./food_sales2.csv")
    df = df.dropna()

    all_dish_id = df['DishID'].unique()
    df_with_id = df
    df_with_id['Vegetarian'] = le.fit_transform(df_with_id['Vegetarian'])
    df_with_id['DayOfWeek'] = le.fit_transform(df_with_id['DayOfWeek'])
    future_df_for_all_dishes = pd.DataFrame(columns=['DishID', 'Total Quantity Sales'])
    next_day_df = pd.DataFrame(columns=['DishID', 'Quantity Sales'])
    for i in all_dish_id:
        dish_1_data = df_with_id[df_with_id['DishID'] == i]
        vegetarian_value = dish_1_data.at[dish_1_data.index[1], 'Vegetarian']
        price_value = dish_1_data.at[dish_1_data.index[1], 'Price']
        dishID_value = dish_1_data.at[dish_1_data.index[1], 'DishID']

        all_transactions = [transaction for date in all_dates for transaction in generate_transactions(date, vegetarian_value, price_value, dishID_value)]
        # all_transactions_df = pd.DataFrame([all_transactions])
        # future_X = pd.DataFrame(columns=['Date','DishID', 'Vegetarian', 'Price', 'DayOfWeek', 'Occasion'])
        future_X = pd.DataFrame(all_transactions, columns=['Date', 'DishID', 'Vegetarian', 'Price', 'DayOfWeek', 'Occasion'])
        # future_X = pd.concat([future_X, all_transactions_df], ignore_index=True)

        future_X.set_index('Date', inplace=True)

        no_of_unique_occasion = future_X['Occasion'].unique()
        dish_1_data = dish_1_data[dish_1_data['Occasion'].isin(no_of_unique_occasion)]
        future_X = pd.get_dummies(future_X, columns=['Occasion'], prefix='Occasion')
        future_X['DayOfWeek'] = le.fit_transform(future_X['DayOfWeek'])

        dish_1_data['Date'] = pd.to_datetime(dish_1_data['Date'])
        dish_1_data = dish_1_data.sort_values(by='Date')

        features = ['DishID', 'Vegetarian', 'Price', 'DayOfWeek', 'Occasion']
        target = 'QuantitySold'

        df_encoded = pd.get_dummies(dish_1_data[features + [target]])

        train_size = int(0.8 * len(df_encoded))
        train, test = df_encoded.iloc[:train_size, :], df_encoded.iloc[train_size:, :]

        X_train, y_train = train.drop(target, axis=1), train[target]
        X_test, y_test = test.drop(target, axis=1), test[target]

        model_rf = RandomForestRegressor()
        model_rf.fit(X_train, y_train)

        y_pred = model_rf.predict(X_test)

        future_y_pred = model_rf.predict(future_X)

        xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
        xgb_model.fit(X_train, y_train)

        y_pred_xgb = model_rf.predict(X_test)

        future_y_pred_xgb = model_rf.predict(future_X)

        ensemble_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

        ensemble_train_data = np.column_stack((y_pred_xgb, y_pred))
        ensemble_model.fit(ensemble_train_data, y_test)

        ensemble_predictions_gbr = ensemble_model.predict(ensemble_train_data)

        ensemble_future_data = np.column_stack((future_y_pred_xgb, future_y_pred))

        future_y_pred_ensemble_gbr = ensemble_model.predict(ensemble_future_data)

        future_results_df_ensemble_gbr = pd.DataFrame({'Predicted': future_y_pred_ensemble_gbr}, index=future_X.index)

        future_results_df_ensemble_gbr['Predicted'] = future_results_df_ensemble_gbr['Predicted'].round().astype(int)
        
        row_next_day = future_results_df_ensemble_gbr.loc[next_day_date]
        
        if not row_next_day.empty:
            next_day_sales = row_next_day['Predicted']

        total_quant = future_results_df_ensemble_gbr["Predicted"].sum()

        add_dish_in_total_pred = {"DishID": i, "Total Quantity Sales": total_quant}
        add_dish_in_total_pred = pd.DataFrame([add_dish_in_total_pred])

        add_dish_in_next_day = {"DishID": i, "Quantity Sales": next_day_sales}
        add_dish_in_next_day = pd.DataFrame([add_dish_in_next_day])

        # future_df_for_all_dishes = future_df_for_all_dishes.append(add_dish_in_total_pred, ignore_index=True)
        future_df_for_all_dishes = pd.concat([future_df_for_all_dishes, add_dish_in_total_pred], ignore_index=True)
        # next_day_df = next_day_df.append(add_dish_in_next_day, ignore_index=True)
        next_day_df = pd.concat([next_day_df, add_dish_in_next_day], ignore_index=True)
        
        json_data_future_df_for_all_dishes = future_df_for_all_dishes.to_json(orient='records')
        json_data_next_day_df = next_day_df.to_json(orient='records')
    
    return {"document1": json_data_future_df_for_all_dishes, "document2": json_data_next_day_df}



@app.route('/chart', methods=['GET', 'POST'])
def chart_predict():
    try:
        # Read dataset from a CSV file
        dataset_path = './../src/static/agmarket_dataset.csv'
        dataset = pd.read_csv(dataset_path)

        # Retrieve data from the request (commodity, district, market, and training data)
        data = request.get_json()
        print(data)
        # Get input data from the frontend
        Commodity = int(data['commodity'])
        start_day = int(data['start_day'])
        start_month = int(data['start_month'])
        start_year = int(data['start_year'])
        end_day = int(data['end_day'])
        end_month = int(data['end_month'])
        end_year = int(data['end_year'])
        state = int(data.get('state'))  # Default value is 1, update as needed
        district = int(data.get('district'))  # Default value is 17, update as needed
        market = int(data.get('market'))  # Default value is 109, update as needed

        # Create a start date and end date object
        start_date = datetime(start_year, start_month, start_day)
        end_date = datetime(end_year, end_month, end_day)
        state_name = state
        district_name = district
        market_center_name = market

        # Initialize an empty list to store predictions
        predictions = []

        # Loop through the date range and make predictions
        while start_date <= end_date:
            # Extract relevant features from the current date
            Commodity = Commodity
            day = start_date.day
            month = start_date.month
            year = start_date.year

            # Filter the training data based on the selected commodity, district, and market
            selected_data = dataset[(dataset['Commodity'] == Commodity) &
                                    (dataset['District'] == district_name) &
                                    (dataset['Market'] == market_center_name)]

            # Check if there is data to train the models
            if selected_data.empty:
                return jsonify({'error': 'No data available for the specified conditions'})

            # Feature selection
            selected_features = selected_data[['Day', 'Month', 'Year']]
            target = selected_data[['MODAL', 'MIN', 'MAX']]

            # Train Random Forest model
            rf_model = RandomForestRegressor()
            rf_model.fit(selected_features, target)

            # Train XGBoost model
            xgb_reg = XGBRegressor(random_state=42)
            xgb_reg.fit(selected_features, target)

            # Save the trained models (you might want to use a more robust serialization method)
            joblib.dump(rf_model, 'rf_model.joblib')
            joblib.dump(xgb_reg, 'xgb_model.joblib')

            # Perform predictions using your model
            feature_values = [day, month, year]
            prediction_rf = rf_model.predict([feature_values])
            prediction_xgb = xgb_reg.predict([feature_values])

            # Append the prediction to the list
            predictions.append({
                'date': start_date.strftime('%d-%m-%Y'),
                'modal': (prediction_rf[0][0] + prediction_xgb[0][0]) / 2,
                'min': (prediction_rf[0][1] + prediction_xgb[0][1]) / 2,
                'max': (prediction_rf[0][2] + prediction_xgb[0][2]) / 2
            })

            # Increment the date by one day
            start_date += timedelta(days=1)

        # Construct the response with predictions
        response = {'predictions': predictions}
        return jsonify(response)

    except Exception as e:
        # Handle exceptions
        error_response = {
            'error_message': str(e)
        }
        return jsonify(error_response), 400
    
@app.route('/predict', methods=['GET', 'POST'])
def predict_price():

    try:
        # Read dataset from a CSV file 
        dataset_path = './../src/static/agmarket_dataset.csv'
        dataset = pd.read_csv(dataset_path)
        print(dataset)

        # Retrieve data from the request (commodity, district, market, and training data)
        data = request.get_json()

        commodity = int(data['Commodity'])
        district = int(data['district_name'])
        market = int(data['market_center_name'])
        day = int(data['day'])
        month = int(data['month'])
        year = int(data['year'])
        # training_data = pd.DataFrame(data['training_data'])

        # Filter the training data based on the selected commodity, district, and market
        selected_data = dataset[(dataset['Commodity'] == int(commodity)) & 
                                    (dataset['District'] == int(district)) & 
                                    (dataset['Market'] == int(market))]
        
        print("Unique Commodity values in the dataset:", dataset['Commodity'].unique())

        print("Selected Commodity value:", commodity)

        print(selected_data)
    

        # Check if there is data to train the models
        if selected_data.empty:
            return jsonify({'error': 'No data available for the specified conditions'})

        # Feature selection
        selected_features = selected_data[[ 'Day', 'Month', 'Year']]
        target = selected_data[['MODAL','MIN', 'MAX']]

        # Train Random Forest model
        rf_model = RandomForestRegressor()
        rf_model.fit(selected_features, target)

        # Train XGBoost model
        xgb_reg = XGBRegressor(random_state=42)
        xgb_reg.fit(selected_features, target)

        # Save the trained models (you might want to use a more robust serialization method)
        joblib.dump(rf_model, 'rf_model.joblib')
        joblib.dump(xgb_reg, 'xgb_model.joblib')

        # feature_values = [Commodity, state_name, district_name, market_center_name,Variety, group_name, Arrival, day, month, year]
        input_data = pd.DataFrame({'Day': day, 'Month': month, 'Year': year} , index=[0])
        
        rf_prediction = rf_model.predict(input_data)
        print(rf_prediction)
        xgb_prediction = xgb_reg.predict(input_data)
        print(xgb_prediction)

            # Construct the response with the prediction result
        response = {
                'modal': (rf_prediction[0][0] + xgb_prediction[0][0]) / 2,
                'min': (rf_prediction[0][1] + xgb_prediction[0][1]) / 2,
                'max': (rf_prediction[0][2] + xgb_prediction[0][2]) / 2,
                
            }
        return jsonify(response)
    except Exception as e:
            # Handle exceptions
            error_response = {
                'error_message': str(e)
            }
            return jsonify(error_response), 400 

@app.route('/notifs', methods=['GET', 'POST'])
def notifs_predict():
    try:
        # Read dataset from a CSV file 
        dataset_path = './../src/static/agmarket_dataset.csv'
        dataset = pd.read_csv(dataset_path)
        print(dataset)
        # Calculate the end date as 10 days from the current date
        current_date = datetime.now().date()
        end_date = current_date + timedelta(days=3)

        # Initialize an empty list to store predictions
        predictions = []

        # Loop through commodities 1 to 3
        for commodity in range(1, 4):  # Assumes 1 is tomatoes, 2 is onions, 3 is potatoes
            commodity_predictions = []

            # Reset the current_date for each commodity
            current_date = datetime.now().date()

            # Loop through the date range and make predictions
            while current_date <= end_date:
                # Extract relevant features from the current date
                # Modify these as needed to match your dataset
                state_name = 1
                district_name = 1
                market_center_name = 1
                day = current_date.day
                month = current_date.month
                year = current_date.year

                # Filter the training data based on the selected commodity, district, and market
                selected_data = dataset[(dataset['Commodity'] == int(commodity)) & 
                                  (dataset['District'] == int(district_name)) & 
                                  (dataset['Market'] == int(market_center_name))]

                # Perform predictions using your model
                feature_values = [day, month, year]

                print(selected_data)
   

                # Check if there is data to train the models
                if selected_data.empty:
                    return jsonify({'error': 'No data available for the specified conditions'})

                # Feature selection
                selected_features = selected_data[[ 'Day', 'Month', 'Year']]
                target = selected_data[['MODAL','MIN', 'MAX']]

                # Train Random Forest model
                rf_model = RandomForestRegressor()
                rf_model.fit(selected_features, target)

                # Train XGBoost model
                xgb_reg = XGBRegressor(random_state=42)
                xgb_reg.fit(selected_features, target)

                # Save the trained models (you might want to use a more robust serialization method)
                joblib.dump(rf_model, 'rf_model.joblib')
                joblib.dump(xgb_reg, 'xgb_model.joblib')

                prediction_rf = rf_model.predict([feature_values])
                prediction_xgb = xgb_reg.predict([feature_values])

                # Store the prediction in the dictionary
                commodity_predictions.append ({
                    'date': current_date.strftime('%d-%m-%Y'),
                    'modal': (prediction_rf[0][0] + prediction_xgb[0][0]) / 2,
                    'min': (prediction_rf[0][1] + prediction_xgb[0][1]) / 2,
                    'max': (prediction_rf[0][2] + prediction_xgb[0][2]) / 2,
                    'commodity': commodity
                })
                print(commodity_predictions)

               

                # # Append the prediction to the list
                # commodity_predictions.append({
                #     'date': current_date.strftime('%d-%m-%Y'),
                #     'modal': prediction[0][0],
                #     'min': prediction[0][1],
                #     'max': prediction[0][2],
                #     'commodity': commodity
                # })

                # Increment the date by one day
                current_date += timedelta(days=1)

            # Append the commodity predictions to the all_predictions list
            predictions.extend(commodity_predictions)

        # Construct the response with all predictions
        response = {'predictions': predictions}
        return jsonify(response)
    except Exception as e:
        # Handle exceptions
        error_response = {
            'error_message': str(e)
        }
        return jsonify(error_response), 400 

@app.route('/today', methods=['GET','POST'])
def today_price():
    try:
        current_date = datetime.now().date()

         # Read dataset from a CSV file 
        dataset_path = './../src/static/agmarket_dataset.csv'
        dataset = pd.read_csv(dataset_path)
        print(dataset)

        commodities = {
            'Tomato': 1,
            'Potato': 2,
            'Onion': 3,
        }

        # Initialize an empty dictionary to store responses
        predictions = {}
        
        # state_name = 1
        # district_name = 1
        # market_center_name = 1
        day = current_date.day
        month = current_date.month
        year = current_date.year

        for commodity, commodity_value in commodities.items():
            selected_data = dataset[(dataset['Commodity'] == commodity_value) & 
                                      (dataset['District'] == 1) & 
                                      (dataset['Market'] == 1)]
            print(selected_data)

            if not selected_data.empty:
                # Feature selection
                selected_features = selected_data[['Day', 'Month', 'Year']]
                target = selected_data[['MODAL', 'MIN', 'MAX']]

                # Train Random Forest model
                rf_model = RandomForestRegressor()
                rf_model.fit(selected_features, target)

                # Train XGBoost model
                xgb_reg = XGBRegressor(random_state=42)
                xgb_reg.fit(selected_features, target)

                # Save the trained models (you might want to use a more robust serialization method)
                joblib.dump(rf_model, f'rf_model_{commodity}.joblib')
                joblib.dump(xgb_reg, f'xgb_model_{commodity}.joblib')

                # feature_values = [commodity_value, state_name, district_name, market_center_name, Variety, group_name, Arrival, day, month, year]
                input_data = pd.DataFrame({'Day': [day], 'Month': [month], 'Year': [year]}, index=[0])

                rf_prediction = rf_model.predict(input_data)
                print(rf_prediction)
                xgb_prediction = xgb_reg.predict(input_data)
                print(xgb_prediction)

                # Construct the response with the prediction result
                predictions[commodity] = {
                    'modal': (rf_prediction[0][0] + xgb_prediction[0][0]) / 2,
                    'min': (rf_prediction[0][1] + xgb_prediction[0][1]) / 2,
                    'max': (rf_prediction[0][2] + xgb_prediction[0][2]) / 2,
                }

        # Return predictions for all commodities
        return jsonify(predictions)

    except Exception as e:
        # Handle exceptions, e.g., invalid input data
        error_response = {
            'error_message': str(e)
        }
        return jsonify(error_response), 400

@app.route('/compare', methods=['POST'])
def compare_price():
    try:
         # Read dataset from a CSV file
        dataset_path = './../src/static/agmarket_dataset.csv'
        dataset = pd.read_csv(dataset_path)
        data = request.get_json()
        print(data)
        # Extract parameters from the request
        district_name = int(data['district'])
        day = int(data['day'])
        month = int(data['month'])
        year = int(data['year'])
        market_values = data['markets']
        

        # Sample commodities
        commodities = {
            'Onion': 2,
            'Tomato': 1,
            'Potato': 3
        }

        # Initialize an empty dictionary to store responses
        predictions = {}

        for commodity, commodity_value in commodities.items():
            # Assuming your model features are in the following order
            feature_values = [day, month, year]

            # Initialize a sub-dictionary for the current commodity
            predictions[commodity] = {}

            # Loop through market values and make predictions
            for market_value in market_values:
                # Update market center name for each iteration
                # feature_values[3] = market_value

                # Filter the training data based on the selected commodity, district, and market
                selected_data = dataset[(dataset['Commodity'] == int(commodity_value)) &
                                        (dataset['District'] == int(district_name)) &
                                        (dataset['Market'] == int(market_value))]
                print(commodity_value)
                print(selected_data)
                # Check if there is data to train the models
                if selected_data.empty:
                    predictions[commodity][market_value] = {'error': 'No data available for the specified conditions'}
                else:
                    # Feature selection
                    selected_features = selected_data[['Day', 'Month', 'Year']]
                    target = selected_data[['MODAL', 'MIN', 'MAX']]

                    # Train Random Forest model
                    rf_model = RandomForestRegressor()
                    rf_model.fit(selected_features, target)

                    # Train XGBoost model
                    xgb_reg = XGBRegressor(random_state=42)
                    xgb_reg.fit(selected_features, target)

                    # Save the trained models (you might want to use a more robust serialization method)
                    joblib.dump(rf_model, f'rf_model_{commodity}_{market_value}.joblib')
                    joblib.dump(xgb_reg, f'xgb_model_{commodity}_{market_value}.joblib')

                    # Perform predictions using your model
                    prediction_rf = rf_model.predict([feature_values])
                    prediction_xgb = xgb_reg.predict([feature_values])

                    # Store the prediction in the dictionary
                    predictions[commodity][market_value] = {
                        'modal': (prediction_rf[0][0] + prediction_xgb[0][0]) / 2,
                        'min': (prediction_rf[0][1] + prediction_xgb[0][1]) / 2,
                        'max': (prediction_rf[0][2] + prediction_xgb[0][2]) / 2,
                    }
                    print(predictions)

        return jsonify(predictions)

    except Exception as e:
        # Handle exceptions, e.g., invalid input data
        error_response = {
            'error_message': str(e)
        }
        return jsonify(error_response), 400

@app.route("/openai", methods = ['GET', 'POST'])
def openai():
    df = pd.read_csv('./past_month_data.csv')
    llm = OpenAI(
        api_token="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",
        temperature=0.7
    )
    sdf = SmartDataframe(df,config={"llm":llm,"enforce_privacy":True})
    result = sdf.chat('suggest some coupons / offers for all dish such that my sellingPrice should not go below the costPrice in a sentence and append it in dataset')
    print(result)
    no_of_unique_dish = df["dish_name"].nunique()
    top_5_rows = result.head(no_of_unique_dish)
    coupons = top_5_rows[["dish_name", "offer"]]
    coupon_json_data = coupons.to_json(orient='records')
    print(coupon_json_data)
    return coupon_json_data, 200

if __name__ == "__main__":
    app.run(debug=True)
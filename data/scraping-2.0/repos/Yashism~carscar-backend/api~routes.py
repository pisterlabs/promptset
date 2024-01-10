import uuid
from werkzeug.utils import secure_filename
from flask import jsonify, request, Blueprint, send_file, redirect, url_for, make_response
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from bson import ObjectId
import gridfs
import requests
import base64
from io import BytesIO
import openai
import json
from . import api_bp
import os
import re
openai.api_key = os.environ.get('OPENAI_API_KEY')

UPLOAD_FOLDER = './'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

uri = "mongodb+srv://yashw:Hackuta2023@carscar.dfpzhz4.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['carscar']
fs = gridfs.GridFS(db)
car_collection = db['car_collection']
analysis_collection = db['analysis_collection']

api_key = os.environ.get('API_KEY')


@api_bp.route('/api/addCar', methods=['POST', 'GET'])
def add_car():
    print("Add Car Called")
    print("open ai key", os.environ.get('OPENAI_API_KEY'))
    report_id = request.form.get('report_id')
    # Get the VIN number from the form
    vin_number = request.form.get('vin_number')
    print("VIN Number : ", vin_number)

    if vin_number:  # If VIN number is provided
        # Call your get_VIN_infoV2 function to get the vehicle info
        vehicle_info = get_VIN_infoV2(vin_number)
        make = vehicle_info.get('make', 'N/A')
        model = vehicle_info.get('model', 'N/A')
        year = vehicle_info.get('year', 'N/A')
    else:  # If VIN number is not provided
        name = request.form.get('name')
        make = request.form.get('make')
        model = request.form.get('model')
        year = request.form.get('year')

    image_base64s = []

    for img in request.files.getlist('images'):
        base64_image = add_car_to_db_and_get_base64(img)
        image_base64s.append(base64_image)

    report_id = str(uuid.uuid4())

    car_data = {
        'report_id': report_id,
        'make': make,
        'model': model,
        'year': year,
        'image_base64s': image_base64s,
        'JSON': [],
        'AI_estimated_cost': 0,
        'parts_toBeReplaced': [],
        'scratch_cost': 0,
        'replacement_price': 0,
        'da_cost': 0
    }

    result = car_collection.insert_one(car_data)

    # if result.acknowledged:
    #     return redirect(url_for('api_bp.get_damage_analysis',_id=result.inserted_id))

    # else:
    #     return jsonify(error="Failed to store car details.")

    if result.acknowledged:
        # Directly call get_damage_analysis and pass the _id argument
        # return redirect(url_for('api_bp.getreport', _id=result.inserted_id))
        status, response_content = get_damage_analysis(result.inserted_id)

        print("status : ", status)
        print("response content : ", type(response_content))
        if response_content == 200:
            # Redirect to the getreport endpoint
            get_report_response = getreport(result.inserted_id)
            print("Get Report back : ", get_report_response)
            return jsonify(get_report_response)
        else:
            print("fucked in the data analysis")
            return jsonify(error="Failed to get report analysis.")
    else:
        return jsonify(error="Failed to store car details.")


@api_bp.route('/api/getImage/<string:image_id>', methods=['GET'])
def get_image(image_id):
    try:
        oid = ObjectId(image_id)
        grid_out = fs.get(oid)
        response = send_file(BytesIO(grid_out.read()), mimetype='image/jpeg',
                             as_attachment=True, attachment_filename=grid_out.filename)
        return response
    except Exception as e:
        return jsonify(error=str(e))


def add_car_to_db_and_get_base64(img):
    filename = secure_filename(img.filename)
    image_id = fs.put(img, filename=filename)
    grid_out = fs.get(image_id)
    image_content = grid_out.read()
    base64_image = base64.b64encode(image_content).decode('utf-8')
    return base64_image


@api_bp.route('/api/get_VIN_info/<string:VIN>', methods=['GET'])
def get_VIN_info(VIN):
    url = f'https://auto.dev/api/vin/{VIN}?apikey={api_key}'
    headers = {
        'Authorization': f'Bearer {api_key}',
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        response_data = response.json()
        damage_response = get_damage_analysis(response_data)
        if damage_response.status_code == 200:
            return jsonify(response_data), 200
    else:
        return jsonify({'error': f'Failed to retrieve data: {response.status_code}'}), 500


@api_bp.route('/api/get_VIN_infoV2', methods=['GET', 'POST'])
def get_VIN_infoV2(vin_num):
    url = f'https://auto.dev/api/vin/{vin_num}?apikey={api_key}'
    headers = {
        'Authorization': f'Bearer {api_key}',
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        response_data = response.json()

        # Extracting required values from the response
        year = response_data['years'][0]['year'] if 'years' in response_data and response_data['years'] else 'N/A'
        make_name = response_data['make']['name'] if 'make' in response_data and 'name' in response_data['make'] else 'N/A'
        model_name = response_data['model']['name'] if 'model' in response_data and 'name' in response_data['model'] else 'N/A'

        vehicle_info = {
            'year': year,
            'make': make_name,
            'model': model_name
        }

        print("Vehicle Info : ", vehicle_info)
        return vehicle_info
    else:
        return jsonify({'error': f'Failed to retrieve data: {response.status_code}'}), 500


@api_bp.route('/api/getDamageAnalysis/<string:_id>')
def get_damage_analysis(_id):
    oid = ObjectId(_id)
    car_doc = car_collection.find_one({'_id': oid})
    if car_doc is None:
        return jsonify({'error': 'Car not found'}), 404

    json_responses = []
    for base64_image in car_doc['image_base64s']:
        # Convert Base64 to image URL if needed or use it directly
        # Here, assuming you can use base64 data directly
        data = {
            'draw_result': False,
            'remove_background': False,
            'image': base64_image
        }
        headers = {
            'content-type': 'application/json',
            'X-RapidAPI-Key': api_key,
            'X-RapidAPI-Host': 'vehicle-damage-assessment.p.rapidapi.com'
        }
        response = requests.post(
            'https://vehicle-damage-assessment.p.rapidapi.com/run',
            json=data,
            headers=headers
        )
        print(response.status_code)
        if response.status_code == 200:
            json_responses.append(response.json())
        else:
            return jsonify({'error': 'Failed to get damage analysis'}), 500

    try:
        car_collection.update_one(
            {"_id": oid},
            {"$set": {"JSON": json_responses}}
        )
        return jsonify({'message': 'Damage analysis successful'}), 200
    except:
        return jsonify({'error': 'DB update unsuccessful'}), 400


def parse_ai_response(response_data):
    print("PARSEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE AIIIIIIIIIIIIIIIIIIIIIII RESPONSEEEEEEEEEEEEEEEEEEEEEEEEE")
    cost_re = re.compile(r'AI_estimated_cost: \$(\d+)', re.IGNORECASE)
    parts_re = re.compile(r'parts_toBeReplaced: ([\w, ]+)', re.IGNORECASE)
    scratch_re = re.compile(r'scratch_cost: \$(\d+)', re.IGNORECASE)
    replacement_re = re.compile(r'replacement_price: \$(\d+)', re.IGNORECASE)

    try:
        cost_match = cost_re.search(response_data)
        if cost_match is None:
            print("No match found for AI_estimated_cost")
            raise ValueError("Failed to parse AI_estimated_cost")

        parts_match = parts_re.search(response_data)
        if parts_match is None:
            print("No match found for parts_toBeReplaced")
            raise ValueError("Failed to parse parts_toBeReplaced")

        scratch_match = scratch_re.search(response_data)
        if scratch_match is None:
            print("No match found for scratch_cost")
            raise ValueError("Failed to parse scratch_cost")

        replacement_match = replacement_re.search(response_data)
        if replacement_match is None:
            print("No match found for replacement_price")
            raise ValueError("Failed to parse replacement_price")

        AI_estimated_cost = int(cost_match.group(1))
        parts_toBeReplaced = parts_match.group(1).split(', ')
        scratch_cost = int(scratch_match.group(1))
        replacement_price = int(replacement_match.group(1))

    except Exception as e:
        print(e)
        raise  # re-raise the exception to ensure it's handled by the calling code

    return AI_estimated_cost, parts_toBeReplaced, scratch_cost, replacement_price


def get_averages(costs, scratch_costs, replacement_prices):
    # Calculate averages
    avg_cost = sum(costs) / (len(costs) if costs else 1)
    avg_scratch_cost = sum(scratch_costs) / \
        (len(scratch_costs) if scratch_costs else 1)
    avg_replacement_price = sum(
        replacement_prices) / (len(replacement_prices) if replacement_prices else 1)

    return avg_cost, avg_scratch_cost, avg_replacement_price


@api_bp.route('/api/getreport/<string:_id>')
def getreport(_id):
    # Retrieve elements from the database using _id
    car_doc = car_collection.find_one({'_id': ObjectId(_id)})
    if car_doc is None:
        return jsonify({'error': 'Car not found'}), 404

    make = car_doc['make']
    model = car_doc['model']
    year = car_doc['year']
    output = []
    for i in car_doc['JSON']:
        current_output = i['output']
        output.append(current_output)
    # Uncomment the line below if you want to use a hardcoded output value
    # output = {'elements': [{'bbox': [126, 71, 162, 90], 'damage_category': 'severe_scratch', 'damage_color': [0, 0, 120], 'damage_id': '2', 'damage_location': 'right_rear_door', 'score': 0.631511}, {
    #     'bbox': [55, 19, 189, 98], 'damage_category': 'severe_deformation', 'damage_color': [0, 95, 255], 'damage_id': '5', 'damage_location': 'right_rear_door', 'score': 0.885108}], 'object_id': 'base64'}
    print("Output Array : ", output)

    # Structuring the message to guide the AI
    messages = [
        {
            "role": "user",
            "content": f"This is a {year} {make} {model}. It has been in an accident and here are the damages I found: {output}."
        },
        {
            "role": "system",
            "content": "You are a virtual auto repair estimator capable of providing detailed cost estimates based on damage descriptions."
        },
        {
            "role": "user",
            "content": (
                "Provide a cost estimate for the mentioned damages. The response must be formatted strictly as 'Field: Value' on separate lines, with no additional text or explanations. If the format is not followed, this conversation will be terminated. Assume average cost for parts and labor in the US as of 2023. Each field should be followed immediately by its value, and each line should end with a newline character and nothing else:\n"
                "1. AI_estimated_cost: $\n"
                "2. parts_toBeReplaced:\n"
                "3. scratch_cost: $\n"
                "4. replacement_price: $\n"
            )
        }
    ]

    AI_estimated_cost, parts_toBeReplaced, scratch_cost, replacement_price = None, None, None, None

    while None in [AI_estimated_cost, parts_toBeReplaced, scratch_cost, replacement_price]:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            max_tokens=1000,
            temperature=1.2,
            messages=messages
        )
        print(response["choices"][0]["message"]["content"])
        response_data = response["choices"][0]["message"]["content"]

        try:
            AI_estimated_cost, parts_toBeReplaced, scratch_cost, replacement_price = parse_ai_response(
                response_data)
        except ValueError as e:
            print(f'Error: {str(e)}, retrying...')  # Print the error and retry

    # Update the database with the parsed values
    car_collection.update_one(
        {'_id': ObjectId(_id)},
        {
            '$set': {
                'AI_estimated_cost': AI_estimated_cost,
                'parts_toBeReplaced': parts_toBeReplaced,
                'scratch_cost': scratch_cost,
                'replacement_price': replacement_price,
            }
        }
    )

    # Call pdf_gen
    # pdf_gen(_id)

    return {"data": {"AI Estimated Cost ": AI_estimated_cost,
                     "Parts to be replaced": parts_toBeReplaced,
                     "Scratches Cost": scratch_cost,
                     "Replacement Price": replacement_price,
                     "make": make,
                     "model": model,
                     "year": year}}


# @api_bp.route('/api/pdf_gen/<string:_id>')
# def pdf_gen(_id):
#     car_doc = car_collection.find_one({'_id': ObjectId(_id)})
#     if car_doc is None:
#         return jsonify({'error': 'Car not found'}), 404
#     make = car_doc['make']
#     model = car_doc['model']
#     year = car_doc['year']
#     AI_estimated_cost = car_doc['AI_estimated_cost']
#     parts_toBeReplaced = car_doc['parts_toBeReplaced']
#     scratch_cost = car_doc['scratch_cost']
#     replacement_price = car_doc['replacement_price']
#     da_cost = car_doc['da_cost']
#     output = []
#     for i in car_doc['JSON']:
#         current_output = i['output']
#         output.append(current_output)
#     print("Output Array : ", output)

#     with open('./report/template.tex', 'r') as template_file:
#         template = template_file.read()

#     # Replace placeholders in the template with the extracted information and generated descriptions
#     template = template.replace('Make', make)
#     template = template.replace('Model', model)
#     template = template.replace('Year', year)

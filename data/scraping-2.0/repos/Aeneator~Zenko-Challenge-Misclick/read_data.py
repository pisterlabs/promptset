import json
import openai
import csv

openai.api_key = open("API_KEY.txt", 'r').read()


def get_stand_data():
    shops_list = []
    id_list = []
    csv_file = open("RawFiles/export_stands20230922 (1).csv")
    headers = next(csv_file)[:-1].split(';')
    for row in csv_file:
        row_data = row[:-1].split(';')
        temp = {}
        for i in range(1, len(headers)):
            temp[headers[i]] = (row_data[i] if ', ' not in row_data[i] else row_data[i].split(', ')) if row_data[i] != '' else None
        shops_list.append(temp)
        id_list.append(row_data[0])

    json_file = open('RawFiles/fdv_stands20230920.geojson', "r")
    stand_list = json.loads(json_file.read())['features']

    for stand in stand_list:
        try:
            stand['properties']['details'] = shops_list[id_list.index(stand['properties']['numero'])]
        except ValueError:
            continue

    json_object = json.dumps({'stand_list': stand_list}, indent=4)
    with open("DataFiles/stand_data.json", "w") as outfile:
        outfile.write(json_object)


def get_location_lists():
    json_file = open("DataFiles/stand_data.json", "r")
    stand_list = json.loads(json_file.read())['stand_list']
    beverages = []
    foods = []
    urgency = []
    stages = []
    toilets = []
    buses = []
    trains = []
    recycle = []
    streets = []
    other = []
    for stand in stand_list:
        ok = False
        if 'details' in stand['properties']:
            if stand['properties']['details']['food_types'] is not None:
                foods.append(stand)
                ok = True
            if stand['properties']['details']['drink_categories'] is not None:
                beverages.append(stand)
                ok = True
        if 'eau' in stand['properties']['numero']:
            beverages.append(stand)
            ok = True
        if 'GSN' in stand['properties']['numero']:
            urgency.append(stand)
            ok = True
        if 'voirie' in stand['properties']['numero']:
            streets.append(stand)
            ok = True
        if 'TransN' in stand['properties']['numero']:
            trains.append(stand)
            ok = True
        if 'scÃ¨ne' in stand['properties']['numero']:
            stages.append(stand)
            ok = True
        if 'camion' in stand['properties']['numero']:
            buses.append(stand)
            ok = True
        if 'WC' in stand['properties']['numero']:
            toilets.append(stand)
            ok = True
        if 'Centre tri' in stand['properties']['numero']:
            recycle.append(stand)
            ok = True
        if ok is False:
            other.append(stand)
    return beverages, foods, urgency, stages, toilets, buses, trains, recycle, streets, other


def get_pins(loc_list):
    return [{"coordinates": [loc['properties']['centerpoint'].split(', ')[1], loc['properties']['centerpoint'].split(', ')[0]],
             "popupText": loc_list.index(loc)} for loc in loc_list]


def get_route_info():
    routes_info = "Routes are structured as follows: each route can be found encapsulated between \"\", and follow this structure "
    with open('DataFiles/Routes.csv', newline='') as file:
        csvfile = csv.reader(file, delimiter=',', quotechar='|')
        for row in csvfile:
            if routes_info[-1] is '\"':
                routes_info += ', \"'
            routes_info += row.__str__() + ('\"' if routes_info[-1] is not ' ' else '')
    routes_info += '. The following information is regarding closed routes, each of them encapsulated between \"\", and follow this structure '
    with open('DataFiles/Modified Routes.csv', newline='') as file:
        csvfile = csv.reader(file, delimiter=',', quotechar='|')
        for row in csvfile:
            if routes_info[-1] is '\"':
                routes_info += ', \"'
            routes_info += row.__str__() + ('\"' if routes_info[-1] is not ' ' else '')

    return routes_info


def get_recycle_info():
    recycle_info = "Recycling rules are split as follows: each of them encapsulated between \"\", and follow this structure "
    with open('DataFiles/Recycle.csv', newline='') as file:
        csvfile = csv.reader(file, delimiter=',', quotechar='|')
        for row in csvfile:
            if recycle_info[-1] is '\"':
                recycle_info += ', \"'
            recycle_info += row.__str__() + ('\"' if recycle_info[-1] is not ' ' else '')

    return recycle_info

"""
OpenAir file operations

- AirScore -
Stuart Mackintosh - Antonio Golfari
2019

"""

import json
import re
from pathlib import Path

import Defines
import folium
import jsonpickle
from aerofiles import openair
from geo import create_arc_polygon
from mapUtils import get_airspace_bbox

NM_in_meters = 1852.00
Ft_in_meters = 0.3048000
hPa_in_feet = 27.3053
colours = {
    'P': '#d42c31',
    'D': '#d42c31',
    'R': '#d42c31',
    'GP': '#d42c31',
    'C': '#d42c31',
    'Z': '#d42c31',
    'CTR': '#d42c31',
    'Q': '#d42c31',
}


def read_openair(filename):
    """reads openair file using the aerofiles library.
    returns airspaces object (openair.reader)"""
    space = None
    airspace_path = Defines.AIRSPACEDIR
    fullname = Path(airspace_path, filename)
    with open(fullname) as fp:
        reader = openair.Reader(fp)

        airspace_list = []
        for record, error in reader:
            if error:
                raise error  # or handle it otherwise
            airspace_list.append(record)
    return airspace_list


def write_openair(data, filename):
    """writes file into airspace folder. No checking if file data is valid openair format.
    returns airspaces object (openair.reader)"""
    space = None
    airspace_path = Defines.AIRSPACEDIR
    # airspace_path = '/home/stuart/Documents/projects/Airscore_git/airscore/airscore/data/airspace/openair/'
    fullname = Path(airspace_path, filename)
    with open(fullname, 'w') as fp:
        fp.write(data)


def airspace_info(record):
    """Creates a dictionary containing details on an airspace for use in front end"""
    return {
        'name': record['name'],
        'class': record['class'],
        'floor_description': record['floor'],
        'floor': convert_height(record['floor'])[1],
        'floor_unit': convert_height(record['floor'])[2],
        'ceiling_description': record['ceiling'],
        'ceiling': convert_height(record['ceiling'])[1],
        'ceiling_unit': convert_height(record['ceiling'])[2],
    }


def convert_height(height_string):
    """Converts feet in metres, GND into 0. leaves FL essentialy untouched. returns a string that can be used in
    labels etc such as "123 m", a int of height such as 123 and a unit such as "m" """
    info = ''
    meters = None

    if height_string == '0':
        return '0', 0, "m"

    if re.search(r"FL", height_string):
        height = int(re.sub(r"[^0-9]", "", height_string))
        return height_string, height, "FL"

    elif re.search(r"ft", height_string):
        if len(re.sub(r"[^0-9]", "", height_string)) > 0:
            feet = int(re.sub(r"[^0-9]", "", height_string))
            meters = round(feet * Ft_in_meters, 1)
            info = f"{height_string}/{meters} m"

    elif re.search(r"m", height_string) or re.search(r"MSL", height_string):
        if len(re.sub(r"[^0-9]", "", height_string)) > 0:
            meters = int(re.sub(r"[^0-9]", "", height_string))
            info = f"{meters} m"

    elif height_string in ('GND', 'SFC'):
        meters = (
            0  # this should probably be something like -500m to cope with dead sea etc (or less for GPS/Baro error?)
        )
        info = "GND / 0 m"
    else:
        return height_string, None, "Unknown height unit"

    return info, meters, "m"


def circle_map(element, info):
    """Returns folium circle mapping object from circular airspace.
    takes circular airspace as input, which may only be part of an airspace"""
    if element['type'] == 'circle':
        floor, _, _ = convert_height(info['floor'])
        ceiling, _, _ = convert_height(info['ceiling'])
        radius = f"{element['radius']} NM/{round(element['radius'] * NM_in_meters, 1)}m"
        return folium.Circle(
            location=(element['center'][0], element['center'][1]),
            popup=f"{info['name']} Class {info['class']} floor:{floor} ceiling:{ceiling} Radius:{radius}",
            radius=element['radius'] * NM_in_meters,
            color=colours[info['class']],
            weight=2,
            opacity=0.8,
            fill=True,
            fill_opacity=0.2,
            fill_color=colours[info['class']],
        )
    else:
        return None


def circle_check(element, info):
    """Returns circle object for checking igc files from circular airspace.
    takes circular airspace as input, which may only be part of an airspace"""
    if element['type'] == 'circle':

        return {
            'shape': 'circle',
            'location': (element['center'][0], element['center'][1]),
            'radius': element['radius'] * NM_in_meters,
            'floor': info['floor'],
            'floor_unit': info['floor_unit'],
            'ceiling': info['ceiling'],
            'ceiling_unit': info['ceiling_unit'],
            'name': info['name'],
        }
    else:
        return None


def polygon_map(record):
    """Returns folium polygon mapping object from multipoint airspace
    takes entire airspace as input"""
    locations = []
    for element in record['elements']:
        if element['type'] == 'point':
            locations.append(element['location'])
        elif element['type'] == 'arc':
            print(f"{record['name']}: * ARC DETECTED *")
            locations.extend(
                create_arc_polygon(element['center'], element['start'], element['end'], element['clockwise'])
            )

    if not locations:
        return None

    floor, _, _ = convert_height(record['floor'])
    ceiling, _, _ = convert_height(record['ceiling'])

    return folium.Polygon(
        locations=locations,
        popup=f"{record['name']} Class {record['class']} floor:{floor} ceiling:{ceiling}",
        color=colours[record['class']],
        weight=2,
        opacity=0.8,
        fill=True,
        fill_opacity=0.2,
        fill_color=colours[record['class']],
    )


def polygon_check(record, info):
    """Returns polygon object for checking igc files from multipoint airspace
    takes entire airspace as input"""
    locations = []
    for element in record['elements']:
        if element['type'] == 'point':
            locations.append(element['location'])
        elif element['type'] == 'arc':
            locations.extend(
                create_arc_polygon(element['center'], element['start'], element['end'], element['clockwise'])
            )

    if not locations:
        return None

    floor, _, _ = convert_height(record['floor'])
    ceiling, _, _ = convert_height(record['ceiling'])

    return {
        'shape': 'polygon',
        'locations': locations,
        'floor': info['floor'],
        'floor_unit': info['floor_unit'],
        'ceiling': info['ceiling'],
        'ceiling_unit': info['ceiling_unit'],
        'name': info['name'],
    }


def create_new_airspace_file(mod_data):
    airspace_path = Defines.AIRSPACEDIR
    # airspace_path = '/home/stuart/Documents/projects/Airscore_git/airscore/airscore/data/airspace/openair/'
    fullname = Path(airspace_path, mod_data['old_filename'])
    new_file = mod_data['new_filename']

    if new_file[-4:] != '.txt':
        new_file += '.txt'

    with open(fullname, 'r') as file:
        data = file.read()
        for change in mod_data['changes']:
            data = modify_airspace(data, change['name'], change['old'], change['new'])
        for space in mod_data['delete']:
            data = delete_airspace(data, space)
        write_openair(data, new_file)
    return new_file


def delete_airspace(file, spacename):
    """Deletes an airspace from file data. Does not write file to disk
    arguments:
    file - file data
    spacename - name of the airspace"""

    all_spaces = file.split("\n\n")

    for space in all_spaces:
        if space.find(spacename) > -1:
            all_spaces.remove(space)
    return "\n\n".join(all_spaces)


def modify_airspace(file, spacename, old, new):
    """modifies airspace. for changing height data.
    arguments:
    file - file data
    spacename - airspace name
    old - string to be relpaced
    new - string to be inserted"""

    all_spaces = file.split("\n\n")
    for i, space in enumerate(all_spaces):
        if space.find(spacename) > 0:
            all_spaces[i] = space.replace(old, new)

    return "\n\n".join(all_spaces)


def create_airspace_map_check_files(openair_filename):
    """Creates file with folium objects for mapping and file used for checking flights.
    :argument: openair_filename located in AIRSPACEDIR"""

    airspace_path = Defines.AIRSPACEDIR
    openair_fullname = Path(airspace_path, openair_filename)

    with open(openair_fullname) as fp:
        _, airspace_list, mapspaces, checkspaces, bbox = openair_content_to_data(fp)
        save_airspace_map_check_files(openair_filename, airspace_list, mapspaces, checkspaces, bbox)


def read_airspace_map_file(openair_filename):
    """Read airspace map file if it exists. Create if not.
    argument: openair file name
    returns: dictionary containing spaces object and bbox"""
    from pathlib import Path

    mapfile_path = Defines.AIRSPACEMAPDIR

    if openair_filename[-4:] != '.txt':
        mapfile_name = openair_filename + '.map'
    else:
        mapfile_name = openair_filename[:-4] + '.map'

    mapfile_fullname = Path(mapfile_path, mapfile_name)

    # if the file does not exist
    if not Path(mapfile_fullname).is_file():
        create_airspace_map_check_files(openair_filename)

    with open(mapfile_fullname, 'r') as f:
        return jsonpickle.decode(f.read())


def get_airspace_map_from_task(task_id: int) -> dict:
    """get airspace map data from task ID"""
    from db.conn import db_session
    from db.tables import TblTask

    with db_session() as db:
        check = db.query(TblTask.airspace_check, TblTask.openair_file).filter_by(task_id=task_id).one()
        if check and check.airspace_check:
            return read_airspace_map_file(check.openair_file)
    return {}


def read_airspace_check_file(openair_filename):
    """Read airspace check file if it exists. Create if not.
    arguent: openair file name
    returns: dictionary containing spaces object and bbox"""
    from pathlib import Path

    checkfile_path = Defines.AIRSPACECHECKDIR

    if openair_filename[-4:] != '.txt':
        checkfile_name = openair_filename + '.check'
    else:
        checkfile_name = openair_filename[:-4] + '.check'

    checkfile_fullname = Path(checkfile_path, checkfile_name)

    # if the file does not exist
    if not Path(checkfile_fullname).is_file():
        create_airspace_map_check_files(openair_filename)

    with open(checkfile_fullname, 'r') as f:
        return json.loads(f.read())


def in_bbox(bbox, fix):
    if bbox[0][0] <= fix.lat <= bbox[1][0] and bbox[0][1] <= fix.lon <= bbox[1][1]:
        return True
    else:
        return False


def altitude(fix, altimeter):
    """returns altitude of specified altimeter from fix"""
    if altimeter == 'barometric' or altimeter == 'baro/gps':
        if fix.press_alt != 0.0:
            return fix.press_alt
        elif altimeter == 'baro/gps':
            return fix.gnss_alt
        else:
            return 'error - no barometric altitude available'
    elif altimeter == 'gps':
        return fix.gnss_alt
    else:
        raise ValueError(f"altimeter choice({altimeter}) not one of barometric, baro/gps or gps")


def openair_content_to_data(content) -> tuple:
    from itertools import tee

    mapspaces = []
    checkspaces = []
    reader = openair.Reader(content)
    reader, reader_2 = tee(reader)
    bbox = get_airspace_bbox(reader_2)
    airspace_list = []
    record_number = 0

    for record, error in reader:
        if error:
            raise error  # or handle it otherwise
        if record['type'] == 'airspace':
            details = airspace_info(record)
            details['id'] = record_number
            airspace_list.append(details)
            polygon = polygon_map(record)
            if polygon:
                mapspaces.append(polygon)
                checkspaces.append(polygon_check(record, details))
            for element in record['elements']:
                if element['type'] == 'circle':
                    mapspaces.append(circle_map(element, record))
                    checkspaces.append(circle_check(element, details))
            record_number += 1

    return record_number, airspace_list, mapspaces, checkspaces, bbox


def save_airspace_map_check_files(openair_filename, airspace_list, mapspaces, checkspaces, bbox):

    mapfile_path = Defines.AIRSPACEMAPDIR
    checkfile_path = Defines.AIRSPACECHECKDIR

    if openair_filename[-4:] != '.txt':
        mapfile_name = openair_filename + '.map'
        checkfile_name = openair_filename + '.check'
    else:
        mapfile_name = openair_filename[:-4] + '.map'
        checkfile_name = openair_filename[:-4] + '.check'

    mapfile_fullname = Path(mapfile_path, mapfile_name)
    checkfile_fullname = Path(checkfile_path, checkfile_name)

    map_data = {'spaces': mapspaces, 'airspace_list': airspace_list, 'bbox': bbox}
    check_data = {'spaces': checkspaces, 'bbox': bbox}
    with open(mapfile_fullname, 'w') as mapfile:
        mapfile.write(jsonpickle.encode(map_data))
    with open(checkfile_fullname, 'w') as checkfile:
        checkfile.write(json.dumps(check_data))

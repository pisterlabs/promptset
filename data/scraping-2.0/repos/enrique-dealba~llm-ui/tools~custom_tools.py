from langchain.agents import Tool
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

## Skyfield API
from skyfield.api import load, Topos
from geopy.geocoders import Nominatim

def ask_user(input: str = ""):
    return "Please provide more information."

ask_user_tool = Tool(
    name="Final Answer",
    func=ask_user,
    description="Useful for when you need to ask the user for more information.",
    )

def get_current_time(*args, **kwargs):
    import subprocess

    result = subprocess.run(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], stdout=subprocess.PIPE)
    current_time = result.stdout.decode('utf-8').strip()
    print(current_time)
    return current_time


def get_skyfield_planets(*args, **kwargs):
    planets = load('de421.bsp')
    names_dict = planets.names()
    flat_list = [name for sublist in names_dict.values() for name in sublist]
    print(flat_list)
    return flat_list

class PlanetDistance(BaseModel):
    planet_names: str = Field(description="Valid planet names from Skyfield")
    # planet_2: str = Field(description="Valid planet name from Skyfield")

def get_planet_distance(planet_names: str,
                        #planet_2: str,
                        *args, **kwargs):
    print(f"planets: {planet_names}")
    planet_list = planet_names.split(", ")
    assert len(planet_list) >= 2 # Make sure there's at least 2 planet entries
    planet_1 = planet_list[0]
    planet_2 = planet_list[1]

    planets = load('de421.bsp')
    # names_dict = planets.names()  # Replace this with your actual call to planets.names()
    # flat_list = [name for sublist in names_dict.values() for name in sublist]
    # assert planet_1 in flat_list
    # assert planet_2 in flat_list
    planet_a = planets[str(planet_1)]
    planet_b = planets[str(planet_2)]
    
    # Create a timescale and ask the current time.
    ts = load.timescale()
    t = ts.now()

    astrometric = planet_a.at(t).observe(planet_b)
    ra, dec, distance = astrometric.radec()
    print(distance)
    return distance

class LatitudeLongitude(BaseModel):
    location: str = Field(description="Valid location like a city name, state, or island")

def get_latitude_longitude(location: str,  *args, **kwargs):
    geolocator = Nominatim(user_agent="YourAppNameHere")
    location = geolocator.geocode(str(location))
    latitude, longitude = location.latitude, location.longitude
    print((latitude, longitude))
    return latitude, longitude

def get_skyfield_satellites(*args, **kwargs):
    stations_url = 'http://celestrak.com/NORAD/elements/stations.txt'
    satellites = load.tle_file(stations_url)
    by_name = {sat.name: sat for sat in satellites}
    print(list(by_name.keys()))
    return list(by_name.keys())

class VisibilityTime(BaseModel):
    satellite_and_location: str = Field(description="Satellite name and valid location")

def get_next_visible_time_for_satellite(satellite_and_location: str, *args, **kwargs):
    print(f"satellite_and_location: {satellite_and_location}")
    sat_loc_list = satellite_and_location.split(" and ")
    assert len(sat_loc_list) >= 2 # Make sure there's at least 2 entries
    satellite_name = sat_loc_list[0]
    location_name = sat_loc_list[1]
    print("location_name:", location_name, len(location_name))
    print("satellite_name:", satellite_name, len(satellite_name))
    lat_, long_ = get_latitude_longitude(location=location_name)

    ts = load.timescale()
    t = ts.now()

    stations_url = 'http://celestrak.com/NORAD/elements/stations.txt'
    satellites = load.tle_file(stations_url)
    by_name = {sat.name: sat for sat in satellites}
    satellite = by_name[str(satellite_name)]
    topos = Topos(latitude=lat_, longitude=long_)

    t0 = ts.now()
    t1 = ts.tt(jd=t0.tt + 1.0)  # Next 24 hours
    t, events = satellite.find_events(topos, t0, t1, altitude_degrees=0.0)

    for ti, event in zip(t, events):
        if event == 0:  # rise above 0 degrees
            next_visible_time = ti.utc_strftime('%Y-%m-%d %H:%M:%S UTC')
            print(f"Next Visibility Time: {next_visible_time}")
            return next_visible_time
        
    return "Next Visibility Time: Not Found"

## LLM Tool Wrappers

get_current_time_tool = Tool(
    name="get_current_time",
    func=get_current_time,
    description = (
        "Use this tool to get the current time."
        "Useful when you need to fill in JSON fields like:"
        "start_time, objective_start_time."
        ),
    )

get_skyfield_planets_tool = Tool(
    name="get_skyfield_planets",
    func=get_skyfield_planets,
    description = (
        "Use this tool to get the list of valid planets from Skyfield."
        "Useful when you need to check if a planet is available in Skyfield."
        ),
    )

get_skyfield_satellites_tool = Tool(
    name="get_skyfield_satellites",
    func=get_skyfield_satellites,
    description = (
        "Use this tool to get the list of valid satellites from Skyfield."
        "Useful when you need to check if a satellite is available in Skyfield."
        ),
    )

get_planet_distance_tool = Tool(
    name="get_planet_distance",
    func=get_planet_distance,
    description = (
        "Use this tool to get distance between planets from Skyfield."
        "Note: Make sure to input valid planet names."
        ),
    args_schema = PlanetDistance
    )

get_latitude_longitude_tool = Tool(
    name="get_latitude_longitude",
    func=get_latitude_longitude,
    description = (
        "Use this tool to get latitude and longitude of a location."
        "Note: Make sure to input valid locations like a city name, state, or island."
        ),
    args_schema = LatitudeLongitude
    )

get_next_visible_time_for_satellite_tool = Tool(
    name="get_next_visible_time_for_satellite",
    func=get_next_visible_time_for_satellite,
    description = (
        "Use this tool to get the next time the user provided satellite will be visible from a given location."
        "Note: Make sure to input a valid satellite name and locations like a city name, state, or island."
        ),
    args_schema = VisibilityTime
    )

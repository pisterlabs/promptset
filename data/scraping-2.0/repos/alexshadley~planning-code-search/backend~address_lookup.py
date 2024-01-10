import os
from typing import List

import dotenv
import geopandas as gpd
import pandas as pd
from geocodio import GeocodioClient
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from shapely import wkt, Point

dotenv.load_dotenv()


def to_geo(df, geo_col):
    df[geo_col] = df[geo_col].apply(wkt.loads)
    return gpd.GeoDataFrame(df, geometry=geo_col)


def load_geo_from_csv(file_name, geo_col):
    df = pd.read_csv(file_name)
    df[geo_col] = df[geo_col].apply(wkt.loads)
    df = gpd.GeoDataFrame(df, geometry=geo_col)

    return df


def get_client():
    return GeocodioClient(os.getenv('GEOCODIO_API_KEY'))

script_dir = os.path.dirname(os.path.abspath(__file__))


prc = pd.read_csv(
    script_dir + "/../sf_parcels.csv.gz",
    usecols=["from_address_num", "to_address_num", "street_name", "shape"],
)
prc = to_geo(prc, "shape")
zn = load_geo_from_csv(script_dir + "/../sf_zoning.csv.gz", "the_geom")


model = ChatOpenAI(model="gpt-3.5-turbo-1106")


class Address(BaseModel):
    number: int = Field(description="the house number of the address")
    street: str = Field(
        description="the street the address is on, in all caps, without any suffix like st or rd"
    )


class AddressResponse(BaseModel):
    addresses: List[Address] = Field(description="all addresses found in the query")


def extract_addresses(query: str) -> List[Address]:
    template_str = """
    Extract addresses from the following query.

    {query}

    {format_instructions}
    """

    parser = PydanticOutputParser(pydantic_object=AddressResponse)
    prompt = PromptTemplate(
        template=template_str,
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    prompt_and_model = prompt | model
    output = prompt_and_model.invoke(
        {
            "query": query,
        }
    )
    response = parser.invoke(output)

    return [a for a in response.addresses if str(a.number) in query]


def get_data_for_addresses(addresses: List[Address]):
    geocoded_addresses = []

    for addy in addresses:
        api_results = get_client().geocode(str(addy.number) + ' ' + addy.street + ' San Francisco, CA')
        geocoded_addresses.append(api_results.get('results')[0])

    gdf = gpd.GeoDataFrame({'Address': [a.get('formatted_address') for a in geocoded_addresses],
                            'geometry': [Point(a.get('location')['lng'],
                                               a.get('location')['lat']) for a in geocoded_addresses]})

    if len(gdf) > 0:
        zoning = gpd.sjoin(gdf, zn, how="inner", predicate="intersects")
        zoning = zoning[['Address', 'zoning', 'districtname']]
        zoning = zoning.rename(columns={'Address': 'address',
                                        'zoning': 'zoning_use_district',
                                        'districtname': 'zoning_use_district_name'})
        return zoning.to_dict('records')


    return []

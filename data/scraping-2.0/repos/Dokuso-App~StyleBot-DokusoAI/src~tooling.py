from typing import Optional, List, Dict
import requests
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.tools import tool
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from IPython.core.display import HTML
from io import BytesIO
from base64 import b64encode
from PIL import Image
import requests
from .schemas import *
from .prompts import prompt_coordination, prompt_combination


baseUrl = 'https://clipfashion-gr7fp45wya-uc.a.run.app/api/v1'

def read_image(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
    except:
        return None
    return img

def display_result(image_batch):
    figures = []
    for img in image_batch:
        img = read_image(img)
        if not img:
            continue
        b = BytesIO()
        img.save(b, format='png')
        figures.append(f'''
            <figure style="margin: 5px !important;">
              <img src="data:image/png;base64,{b64encode(b.getvalue()).decode('utf-8')}" style="width: 90px; height: 120px" >
            </figure>
        ''')
    return HTML(data=f'''
        <div style="display: flex; flex-flow: row wrap; text-align: center;">
        {''.join(figures)}
        </div>
    ''')

@tool
def list_brands():
    """
    List all the brands available in the Dokuso database.

    Returns:
    list: A list of brand names.
    """
    url = f'{baseUrl}/brands_list'
    response = requests.get(url)
    return response.json()['brand_list']


def create_item_data(item):
    """
    Given an item dictionary, this function returns a new dictionary containing only the relevant information
    about the item.

    Parameters:
    item (dict): A dictionary containing all the information about the item.

    Returns:
    dict: A dictionary containing only the relevant information about the item.
    """
    return {
        'itemId': item['id'],
        'name': item['name'],
        'description': item['desc_1'],
        'price': str(round(item['price'], 0))+item['currency'],
        'discount': str(round(item['discount_rate'], 0)*100)+'%',
        'imgUrl': item['img_url'],
        'brand': item['brand'],
        'category': item['category'],
        # 'shopLink': item['shop_link'],
        'shopLink': 'https://dokuso.app/us/en/results/explore/'+'-'.join(item['name'].split(' '))+f'%20{item["id"]}',
        'onSale': item['sale']
    }


@tool(args_schema=SearchItemsQueryInput)
def search_items(query: str, 
                 maxPrice: Optional[float] = None, 
                 category: str = None, 
                 onSale: Optional[bool] = None, 
                 brands: Optional[List[AllowedBrands]] = None,
                 limit: int = 5) -> List[dict]:
    """
    Useful for whe you need to search for items in the Dokuso database using DOKUSO API based on various criteria. All arguments are required.

    Parameters:
    query (str): Natural language query for the search.
    maxPrice (float): Maximum price of items to retrieve.
    category (str): Category of the items ('women', 'men', 'kids', 'home').
    onSale (bool): Whether to search for items on sale.
    brands (list): Brands of the items ('zara', 'massimo dutti', 'mango', 'h&m', 'bershka').
    limit (int: Number of items to retrieve.)

    Returns:
    list: A list of item dictionaries that match the search criteria.
    """

    total_results = []

    params = {
        'query': query,
        'maxPrice': maxPrice,
        'category': category,
        'onSale': onSale,
        'brands':  ','.join([b.name for b in brands]) if brands else None,
        # 'brands':  ','.join(brands) if brands else None,
        'limit': limit
    }
    url = f'{baseUrl}/search'
    response = requests.get(url, params=params)
    if response.status_code == 200:
        results = response.json().get('results', [])
        for item in results:
            item = create_item_data(item)
            total_results.append(item)
    else:
        print(f'Error retrieving results for "{query}"')
        print(response.text)
        
    # display_result(image_batch)
    return total_results

@tool(args_schema=SearchCombinationInput)
def search_combination(userRequest: str, 
                        countQueries: Optional[int] = 3,
                        maxPrice: Optional[float] = None, 
                        category: str = None, 
                        onSale: Optional[bool] = None, 
                        limit: Optional[int] = 5,
                        brands: Optional[List[AllowedBrands]] = None) -> List[dict]:
                        
    """
    Useful for when you need to create a specific style or to complete an outfit requested by the user.

    Parameters:
    userRequest (str): User request.
    countQueries (int): Number of queries to generate.
    maxPrice (float): Maximum price of items to retrieve.
    category (str): Category of the items ('women', 'men', 'kids', 'home').
    onSale (bool): Whether to search for items on sale.
    limit (int): Number of items to retrieve.
    brands (str: Brand of the items ('zara', 'massimo dutti', 'mango', 'h&m', 'bershka').

    Returns:
    list: A list of item dictionaries that match the search criteria.
    """

    total_results = []
    total_images  = []
    queries = generate_fashion_queries_for_combination(userRequest, category, countQueries)
    for q in queries[:4]:
        print(f'Retrieving results for "{q}"')
        params = {
            'query': q,
            'maxPrice': maxPrice,
            'category': category,
            'onSale': onSale,
            'brands':  ','.join([b.name for b in brands]) if brands else None,
            'limit': limit
        }
        url = f'{baseUrl}/search'
        response = requests.get(url, params=params)
        if response.status_code == 200:
            results = response.json().get('results', [])
            for item in results:
                total_images.append(item['img_url'])
                item = create_item_data(item)
                total_results.append(item)
        else:
            print(f'Error retrieving results for "{q}"')
            print(response.text)
    # display_result(total_images)
    return total_results



tagging_functions = [convert_pydantic_to_openai_function(FashionQueriesGenerator)]
    

def generate_fashion_queries_for_combination(userRequest: str, category, count):
    
    model = ChatOpenAI(temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Think carefully, and then tag the text as instructed"),
        ("user", "{input}")
    ])

    desc = prompt_combination(userRequest, category, count)
    tagging_functions[0]['parameters']['properties']['queries']['description'] = desc
    model_with_functions = model.bind(
        functions=tagging_functions,
        function_call={"name": "FashionQueriesGenerator"}
    )


    tagging_chain = prompt | model_with_functions | JsonOutputFunctionsParser()
    response = tagging_chain.invoke(
        {"input": userRequest}
    )
    return response['queries']


@tool(args_schema=ItemId)
def get_product_details(itemId:str)->List[dict]:
    """
    Useful for when you answer questions about a specific product using its item_id.

    Parameters:
    item_id (str): The unique identifier of the item.

    Returns:
    dict: A dictionary containing details of the item.
    """
    url = f'{baseUrl}/product_details?id={itemId}'
    response = requests.get(url)
    return response.json()


@tool(args_schema=ItemId)
def find_similar_products(itemId:str)->List[dict]:
    """
    Useful for when you need to find products similar to the specified itemId in the Dokuso database.

    Parameters:
    itemId (str): The unique identifier of the item to find similar products for.

    Returns:
    list: A list of item dictionaries similar to the specified item.
    """
    url = f'{baseUrl}/most_similar_items?id={itemId}'
    response = requests.get(url)
    return response.json()


@tool(args_schema=CoordinateOutfitInput)
def coordinate_outfit(baseItem: str, 
                      includeAccessories: bool = False, 
                      gender: Optional[str] = None,
                      brands: Optional[List[AllowedBrands]] = None,
                      countOutfits: int = 3,
                      limit: int = 5) -> Dict[str, List[dict]]:
    """
    Provides assistance in coordinating outfits, including matching colors, styles, and accessories.

    Parameters:
    baseItem (str): A base item or color/style to start with for the outfit coordination.
    includeAccessories (bool): Whether to include accessories in the suggestions.
    gender (str): Gender for which the outfit is intended.
    countOutfits (int): Number of outits to suggest.
    limit (int): Number of items to suggest in each outfit.

    Returns:
    List[dict]: A list of coordinated outfit items.
    """

    # Generate fashion queries based on the base item
    queries = generate_fashion_queries_for_coordination(baseItem, includeAccessories, gender, limit)

    # Collect results from all queries
    total_results = {f'outfit {i}': [] for i in range(countOutfits)}
    visited = []
    for query in queries:
        print('Query:', query)
        params = {
            'query': query,
            'category': gender,
            'brands':  ','.join([b.name for b in brands]) if brands else None,
            'limit': countOutfits 
        }
        url = f'{baseUrl}/search'
        response = requests.get(url, params=params)

        if response.status_code == 200:
            results = response.json().get('results', [])
            if results:
                for i in range(countOutfits):
                    item = create_item_data(results[i])
                    total_results[f'outfit {i}'].append(item)
                    visited.append(results[i]['id'])
        else:
            print(f'Error retrieving coordinated items for "{query}"')
            print(response.text)
    
    return total_results


def generate_fashion_queries_for_coordination(baseItem, includeAccessories, gender, limit):
    """
    Generates specific item queries for outfit coordination using OpenAI's API Completion.

    Parameters:
    baseItem (str): The base item or color/style.
    includeAccessories (bool): Whether to include accessories.
    gender (str): Gender for the outfit.
    limit (int): Number of queries to generate.

    Returns:
    List[str]: A list of generated queries.
    """


    model = ChatOpenAI(temperature=0.1) # Adjust temperature as needed for creativity vs relevance
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Think creatively and generate complementary fashion queries."),
        ("user", "{input}")
    ])
    desc = prompt_coordination(baseItem, gender, limit)
    tagging_functions[0]['parameters']['properties']['queries']['description'] = desc
    model_with_functions = model.bind(
        functions=tagging_functions,
        function_call={"name": "FashionQueriesGenerator"}
    )

    tagging_chain = prompt | model_with_functions | JsonOutputFunctionsParser()
    response = tagging_chain.invoke(
        {"input": f"{baseItem} for {gender}"}
    )

    # If includeAccessories is True, append accessory-related queries
    if includeAccessories:
        response['queries'].extend(generate_accessory_queries(limit - len(response['queries'])))

    return response['queries'][:limit]

def generate_accessory_queries(remaining_limit):
    """
    Generates accessory queries.

    Parameters:
    remaining_limit (int): Remaining number of queries to generate.

    Returns:
    List[str]: A list of accessory queries.
    """
    accessories = ['belt', 'watch', 'necklace', 'earrings', 'hat']
    return accessories[:remaining_limit]


@tool(args_schema=StyleDiscoveryInput)
def discover_personal_style(userPreferences: List[str], 
                            lifestyle: str, 
                            favoriteColors: List[str],
                            dislikedItems: List[str] = [],
                            limit: int = 5) -> List[dict]:
    """
    Discovers and suggests personal style options based on user's preferences, lifestyle, and color choices.

    Parameters:
    userPreferences (List[str]): User's style preferences and interests.
    lifestyle (str): The user's lifestyle or typical activities.
    favoriteColors (List[str]): User's favorite colors.
    dislikedItems (List[str]): Items or styles the user dislikes.
    limit (int): Number of style suggestions to retrieve.

    Returns:
    List[dict]: A list of style suggestions that align with the user's preferences.
    """
    
    # Construct the query for style discovery
    query = construct_style_discovery_query(userPreferences, lifestyle, favoriteColors, dislikedItems)
    print(f'Query for style discovery: {query}')
    params = {
        'query': query,
        'limit': limit
    }
    url = f'{baseUrl}/search'
    response = requests.get(url, params=params)

    if response.status_code == 200:
        results = response.json().get('results', [])
        return [create_item_data(item) for item in results]
    else:
        print(f'Error retrieving style suggestions for user preferences')
        print(response.text)
        return []

def construct_style_discovery_query(userPreferences, lifestyle, favoriteColors, dislikedItems):
    """
    Constructs a query for style discovery based on user inputs.

    Parameters:
    userPreferences, lifestyle, favoriteColors, dislikedItems

    Returns:
    str: A query string for style discovery.
    """
    query_elements = userPreferences + [lifestyle] + favoriteColors
    query = ' '.join(query_elements)

    # Adding negative preferences (disliked items) to refine the search
    for item in dislikedItems:
        query += f" -{item}"

    return query
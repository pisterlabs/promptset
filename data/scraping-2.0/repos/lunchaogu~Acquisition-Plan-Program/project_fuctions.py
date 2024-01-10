from openai import OpenAI
import requests
import pandas as pd
from bs4 import BeautifulSoup
from lingua import Language, LanguageDetectorBuilder
import hanzidentifier

def generate_query(category, area):
    unique_items = []
    for item in area:
        if item != '' and item not in unique_items:
            unique_items.append(item)
    unique_items.reverse()
    result = ' '.join(unique_items)
    return(f'{category}, {result}')
#--------------------------------------------------------------------------------------------------------
def get_POI(API_KEY, category, area, pagetoken, url):
    query = generate_query(category, area)
    print(query)
    params = {
        'query': query,
        'key': API_KEY,
        'pagetoken' : pagetoken,
        #'type' : "amusement_park",
        }
    result = requests.get(url, params=params)
    #Convert data into json type first and dataframe then
    result_js = result.json()
    result_df = pd.json_normalize(result_js['results'])
    #Check if there is next page and send the token to params
    if 'next_page_token' in result_js:
        page_token = result_js['next_page_token']
    else:
        page_token = None
    return(result_df, page_token)
#--------------------------------------------------------------------------------------------------------
def cleanup(dataframe, rating_threshold, country):
    dataframe = dataframe.drop_duplicates(subset=['place_id'])
    dataframe = dataframe.drop_duplicates(subset=['formatted_address','name'])
    dataframe = dataframe[dataframe['formatted_address'].str.contains(country)]
    dataframe['country'] = dataframe['plus_code.compound_code'].str.extract(r'((?<=\s)[a-zA-Z]+$)')
    place_info = dataframe['plus_code.compound_code'].str.replace(r'((,)[\sa-zA-Z]+$)', '', regex=True)
    dataframe['state'] = place_info.str.extract(r'((?<=\s)[a-zA-Z]+$)')
    place_info = place_info.str.replace(r'((,)[\sa-zA-Z]+$)', '', regex=True)
    dataframe['city'] = place_info.str.extract(r'((?=\s).+)')
    dataframe = dataframe[['state','city','name','formatted_address', 'place_id', 'rating', 'user_ratings_total', 'types', 'plus_code.compound_code']]
    dataframe = dataframe[dataframe['user_ratings_total'] >= rating_threshold].reset_index(drop=True)
    return(dataframe) 
#---------------------------------------------------------------------------------------------------------
def get_completion(prompt, OpenAI_key, model = 'gpt-3.5-turbo'):
    client = OpenAI(
        api_key=OpenAI_key
        )
    response = client.chat.completions.create(
        messages = [
            {'role': 'user','content': prompt}],
        model=model,
        timeout=20,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message.content
#---------------------------------------------------------------------------------------------------------
#This function is to expand activities by attractions
def explode_AID(AID_df):
    AID_df['Activity_seperated'] = AID_df['response'].apply(
        lambda x: x if pd.isna(x) else x.split(","))
    AID_df = AID_df.explode('Activity_seperated').reset_index(drop=True)
    return(AID_df)

#---------------------------------------------------------------------------------------------------------
headers_klook = {
    'Accept': '*/*',
    'Accept-Language': 'en,zh-CN;q=0.9,zh;q=0.8',
    'Connection': 'keep-alive',
    'Origin': 'https://www.klook.com',
    'Referer': 'https://www.klook.com/',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'cross-site',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
}

headers_traveloca = {
    'authority': 'www.traveloka.com',
    'accept': '*/*',
    'accept-language': 'en,zh-CN;q=0.9,zh;q=0.8',
    'content-type': 'application/json',
    'origin': 'https://www.traveloka.com',
    'referer': 'https://www.traveloka.com/',
    'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'x-domain': 'content',
    'x-route-prefix': 'en-id',
}

headers_klook = {
    'Accept': '*/*',
    'Accept-Language': 'en,zh-CN;q=0.9,zh;q=0.8',
    'Connection': 'keep-alive',
    'Origin': 'https://www.klook.com',
    'Referer': 'https://www.klook.com/',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'cross-site',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
}
#-----------------------------------------------------------------------------------------------------------------
def gmap_mapping(query, url):
    driver.get(url)
    #driver.find_element(By.CSS_SELECTOR,'#searchboxinput').clear()
    #time.sleep(1)
    driver.find_element(By.CSS_SELECTOR,'#searchboxinput').send_keys(query)
    driver.find_element(By.CSS_SELECTOR,'#searchboxinput').send_keys(webdriver.Keys.ENTER)
    time.sleep(2)
    results = []
    try:
        WebDriverWait(driver, 5, 0.5).until(EC.visibility_of_element_located((By.XPATH,'//*[@id="QA0Szd"]/div/div/div[1]/div[2]/div/div[1]/div/div/div[2]/div/div[1]/div[1]/h1')),'No value')
        POI_name = driver.find_element(By.CSS_SELECTOR,'h1').text
        category = driver.find_element(By.CSS_SELECTOR,"#QA0Szd > div > div > div.w6VYqd > div.bJzME.tTVLSc > div > div.e07Vkf.kA9KIf > div > div > div.TIHn2 > div > div.lMbq3e > div.LBgpqf > div > div:nth-child(2) > span > span > button").text
        review = driver.find_element(By.CSS_SELECTOR,"#QA0Szd > div > div > div.w6VYqd > div:nth-child(2) > div > div.e07Vkf.kA9KIf > div > div > div.TIHn2 > div > div.lMbq3e > div.LBgpqf > div > div.fontBodyMedium.dmRWX > div.F7nice > span:nth-child(2) > span > span").text
        review = int(re.sub('[^0-9]','',review))
        result = {
        'POI_name': POI_name,
        'category': category,
        'review': review
        }
        results.append(result)

    except:
        try:
            hotel_path = '//*[@id="QA0Szd"]/div/div/div[1]/div[2]/div/div[1]/div/div/div[2]/div/div[1]/div[2]/div/div/span/span/span/span[2]/span'
            WebDriverWait(driver, 1, 0.2).until(EC.visibility_of_all_elements_located((By.XPATH, hotel_path)),"No value")
            category = driver.find_element(By.XPATH, hotel_path).text
        except:
            try:
                divs = driver.find_elements(By.CSS_SELECTOR,'.THOPZb ')
                for div in divs:
                    POI_name = div.find_element(By.CSS_SELECTOR,'.fontHeadlineSmall').text
                    category = div.find_element(By.CSS_SELECTOR,'div.UaQhfb.fontBodyMedium > div:nth-child(4) > div').text
                    review = div.find_element(By.CSS_SELECTOR,'.UY7F9').text
                    review = int(re.sub('[^0-9]','',review))
                    result = {
                        'POI_name': POI_name,
                        'category': category,
                        'review': review
                        }
                    results.append(result)
                    next
            except:
                result = {
                    'POI_name': None,
                    'category': None,
                    'review': None
                    }
                results.append(result)

    if len(results) == 0:
        results.append({
            'POI_name': None,
            'category': None,
            'review': None
            })
    return(results)
#---------------------------------------------------------------------------------------------------------------------------
#Use embedding and GPT to map
def completion(messages, model, function_data = None):
    completion = None
    print("Calling Open AI")
    if (function_data):
        completion = openai.ChatCompletion.create(
            model=model,
            temperature=0,
            messages=messages,
            function_call={"name":function_data["function_call_component"][0]["name"]},
            functions=function_data["function_call_component"]
        )
    else:
        completion = openai.ChatCompletion.create(
            model=model,
            temperature=0.2,
            messages=messages,
          )

    if (function_data):
        return {
            "content": completion['choices'][0]['message']['function_call']['arguments'],
            "cost": json.dumps(completion)
        }
    else:
        return {
            "content": completion['choices'][0]['message']['content'],
            "cost": json.dumps(completion)
        }
    
#------------------------------------------------------------------------------------------------------------------
def langdetect(text):
    #if text == None:
    #    return '#N/A'
    languages = [Language.ENGLISH, Language.FRENCH, Language.GERMAN, Language.SPANISH, 
                 Language.CHINESE, Language.JAPANESE, Language.THAI, Language.VIETNAMESE,
                 Language.ARABIC,Language.ITALIAN, Language.INDONESIAN, Language.RUSSIAN,
                 Language.SOMALI,Language.CZECH, Language.DANISH, Language.DUTCH, Language.GREEK,
                 Language.HINDI,Language.IRISH, Language.ROMANIAN, Language.SWEDISH,
                 Language.TURKISH, Language.KOREAN
                 ]
    text = text[:50].strip()
    try:
        detector = LanguageDetectorBuilder.from_languages(*languages).build()
        result = str(detector.detect_language_of(text))
        result = result.split('.')[1]
        if result == 'CHINESE':
            if hanzidentifier.has_chinese(text):
                return 'zh-cn' if hanzidentifier.is_simplified(text) else 'zh-Hant'
            else:
                return 'others'
        #print('lang_detector success', result, text )
        return result
    except Exception as error:
        print('lang_detector error', text, error)
        return 'others'
#-------------------------------------------------------------------------------------------------------------------
def lang_process(dataset):
    #dataset = pd.read_csv(route)
    dataset = dataset[~dataset['text'].isna()]
    dataset = dataset.loc[:,['placeId','state','city','website','phone','categoryName','publishedAtDate','text']]
    dataset['publishedAtDate'] = dataset['publishedAtDate'].astype(str)
    dataset['publishedAtDate'] = dataset['publishedAtDate'].apply(lambda x: int(x[:7].replace('-','')))
    #dataset = dataset[(dataset['publishedAtDate'] >= 202301)|(dataset['publishedAtDate'] <= 201912)]
    return dataset
#-------------------------------------------------------------------------------------------------------------------
def review_date_process(dataset):
    #dataset = pd.read_csv(route)
    dataset = dataset[['placeId','state','city','publishedAtDate']]
    dataset = dataset[dataset['publishedAtDate'].notna()]
    dataset['publishedAtDate'] = dataset['publishedAtDate'].apply(lambda x: pd.to_datetime(x[:10]))
    result = dataset.groupby('placeId')['publishedAtDate'].aggregate(['count','max','min'])
    result['review_period'] = result['max']-result['min']
    result['placeId'] = result.index
    result = result.reset_index(drop=True).merge(dataset[['placeId','state','city']], on='placeId', how='left').drop_duplicates(subset='placeId')
    result = result[['placeId','state','city','count','review_period']]
    return result
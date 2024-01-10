from scraper_package import scraper_module
from scraper_package import transform_module

from config import GOOGLE_GEOCODING_API_KEY as goog_access_token
from config import OPENAI_KEY as openai_access_token

races = transform_module.import_not_transformed("sourced_races.json")
costometer = 0
try:
    for i in range(len(races)):
        print(f"""
              TRANSFORMING----------------------
              ----------------------------------
              {races[i]["name"]}
              RACES TRANSFORMED: {i} / {len(races)}
              ----------------------------------
                """)
        ### Check if race already in aborted races
        race_in_aborted = transform_module.return_if_exists(races[i],'transformed_races_aborted.json', 'extract_id')
        if race_in_aborted:
            print("already transformed in aborted run, adding back")
            races[i] = race_in_aborted
            continue
        ### Get website if not allowed or missing
        print("getting website")
        races[i]["website"] = transform_module.check_allowed_url_get_goog(races[i]["website"], races[i]["website_ai_fallback"])
        #races[i]["website"] = transform_module.check_allowed_url_get_goog_selenium(races[i]["website"], races[i]["website_ai_fallback"])
        print("done getting website")
        ### Get website contents
        print("getting website content")
        contents = transform_module.get_website_contents(races[i]["website"])
        races[i]["contents"] = contents
        print("done getting website content")
    
        ###Getting images
        image_search_query = f'"{races[i]["name"]} {transform_module.process_url(races[i]["website"])}"'
        image_url_list = transform_module.get_images_selenium(image_search_query)
        print("got images:")
        print(image_url_list)
        images=[]
        for img in image_url_list:
            images.append(transform_module.convert_and_compress_image(img,max_size_kb=200))
        races[i]["images"] = images 
        print("done with images")
        ### Get website summary
        # Extract specific fields
        title = races[i]['contents']['title']
        description = races[i]['contents']['description']
        h1 = races[i]['contents']['h1']
        paragraphs = races[i]['contents']['p'] if races[i]['contents']['p'] else ""
        h2 = races[i]['contents']['h2'][0] if races[i]['contents']['h2'] else ""
    
        ### Map distances
        races[i]["race_cateogories"] = scraper_module.race_category_mapping(races[i]["distance_m"], races[i]["type"])
        print(races[i]["race_cateogories"])

        # Generate summary prompt
        summary_prompt = f"Given this:\n\nTitle: {title}\nDescription: {description}\nH1: {h1}\nH2: {h2}\n\n Make a summary of the race in Swedish. Pretend that you are a running race director. Write a description in a couple of paragraphs that describes a race like that, you are allowed to freestyle abit. Dont use html elements like Titel:, H1: or H2: in the text. Also emphasize that the race includes the following race categories/distances: {races[i]['race_cateogories']} . And of this type: {races[i]['type']}"
        races[i]["summary"], costometer = transform_module.get_completion(prompt = summary_prompt, costometer=costometer,openai_key=openai_access_token)
    
        if races[i]['summary']:
            short_summary_prompt = f"Given this:{races[i]['summary']} Pretend that you are a running race director and write a shorter summary in Swedish of no more than 500 characters. Also emphasize that the race includes the following race categories/distances: {races[i]['race_cateogories']} . And of this type: {races[i]['type']}"
            races[i]["short_summary"], costometer = transform_module.get_completion(prompt = short_summary_prompt, costometer=costometer,openai_key=openai_access_token)
        else:
            races[i]["short_summary"] = None
        ##location_prompt = f"Given this:\n\nTitle: {title}\nDescription: {description}\nH1: {h1}\nH2: {h2}\n\n Can you make your best guess as to where this race is located? If you can't find it, just write None"
        ##races[i]["ai_place_guess"], costometer = transform_module.get_completion(prompt = location_prompt, costometer=costometer,openai_key=openai_access_token)
        
        name_prompt = f"Given this:\n\nTitle: {title}\nDescription: {description}\nH1: {h1}\nH2: {h2}\n\n Pretend that you are the race director for this running race and give it a name in swedish and using no more than 3 words"
        races[i]["ai_name_guess"], costometer = transform_module.get_completion(prompt = name_prompt, costometer=costometer,openai_key=openai_access_token)
        
    
        ### Get lat and long from google geocoding API using search strings in order of preference
        search_strings = [
            races[i]["place"],
            races[i]["name"],
            races[i]["website"],
            races[i]["website_ai_fallback"],
            races[i]["contents"]["title"],
            races[i]["contents"]["h1"],
            #races[i]["ai_place_guess"],
            races[i]["ai_name_guess"]
        ]
    
        # Get coordinates for the first successful search string
        latitude, longitude = transform_module.get_lat_long_goog(goog_access_token, *search_strings)
        print("done geocoding")
        races[i]["latitude"] = latitude
        races[i]["longitude"] = longitude
    
        # Get counties for found coordinates
        if latitude != 0:
            races[i]["county"] = transform_module.find_county(latitude, longitude)
        else:
            races[i]["county"] = None
            
    print("Exited succesfully")
    print("Costometer: " + str(round(costometer/4000,4)) + "$")
    scraper_module.add_id('transform', races)
    scraper_module.update_source(races,'sourced_races.json','extract_id','is_transformed', True)
    scraper_module.append_or_create_json("transformed_races.json", races)
    print("Stored as transformed_races")
except Exception as e:
    print(f"Exited with error: {e}")
    print("Costometer: " + str(round(costometer/4000,4)) + "$")
    #store away already transformed races
    transformed_races_before_abort = races[:i]
    scraper_module.add_id('transform_abort', transformed_races_before_abort)
    scraper_module.append_or_create_json("transformed_races_aborted.json", transformed_races_before_abort)
    print("Stored as transformed_races_aborted")




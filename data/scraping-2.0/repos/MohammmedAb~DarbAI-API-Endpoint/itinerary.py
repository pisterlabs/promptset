import os,json
import weaviate
from dotenv import load_dotenv
from openai import OpenAI
import random
import time

load_dotenv()

WEAVIATE_CLUSTER_URL = os.environ["WEAVIATE_CLUSTER_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

weaviateClient = weaviate.Client(
    url=WEAVIATE_CLUSTER_URL,
    auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY), 
    additional_headers={"X-OpenAI-Api-Key": OPENAI_API_KEY}
)

client = OpenAI(
    api_key=OPENAI_API_KEY,
)






# This function will return the number of items selected by the user for each category
def num_each_day(selected_tags):
    num_each_day = {}
    hard_coded = {
        "Attractions": 3,
        "Restaurants": 2,
        "Shopping": 0
    }

    total = 0
    for category, num in selected_tags.items():
        total += len(num)
    
    if total == 0:
        return hard_coded
        # print(f"{category}: {len(num)}")
    
    for category, selections in selected_tags.items():
        if len(selections) == 0:
            num_each_day[category] = 0
        else:
            num_each_day[category] = round((len(selections) / total) * 5)
        
    
    return num_each_day


def query_weaviate(category, selections, totalNumberPerCategory, city):
    where_filter = {
    "operator": "Equal",
    "path": ["slugCategoryPOI"],
    "valueText": category
    }

    query_result = weaviateClient.query\
        .get(city, ["name","description", "slugCategoryPOI", "slugCity", "bannerImage","location"])\
        .with_where(where_filter)\
        .with_near_text({"concepts": selections})\
        .with_group_by(properties=["description"],groups= totalNumberPerCategory, objects_per_group= 1)\
        .with_limit(totalNumberPerCategory + 5)\
        .do()

    return query_result

def LLMDescription(LLMPlan):
    assistantId= 'asst_iXLologv6SWugGCWHrOvZ6Iq'  

    thread = client.beta.threads.create()
    # print(thread)
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=str(LLMPlan)
    )

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistantId,
    )

    start_time = time.time()
    timeout = 60

    while run.status == 'in_progress' or run.status == 'queued':
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
        time.sleep(1) 
        print("Waiting for completion...", run.status)
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            print("Operation timed out.")
            return ""

    
    response = client.beta.threads.messages.list(
        thread_id=thread.id,
    )
    
    return response.data[0].content[0].text.value

def LLMCompletions(LLMPlan):
    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "Be an energetic tour guide, summarizing a captivating Saudi Itinerary in a single paragraph. Use vivid language and emojis to showcase the Saudi Arabia's cultural and natural wonders, without lengthy details or day-by-day breakdowns just describe the overall experience in couple of sentences. This should spark excitement and curiosity in the reader to discover more."},
            {"role": "user", "content": str(LLMPlan)}
        ]
    )
    
    try:
        response = completion.choices[0].message.content
    except IndexError:
        response = ""
    except AttributeError:
        response = ""
    
    return response



def itinerary(city, numOfDayes, tags):
    
    num_each_day_counts = num_each_day(tags)

    print("\nNumber of items to be selected each day:")
    for category, num in num_each_day_counts.items():
        print(f"{category}: {num}")

    TripPlan = {}
    for day in range(int(numOfDayes)):
            day_key = "Day " + str(day+1)
            if day_key not in TripPlan:
                TripPlan[day_key] = []

    LLMPlan = {}
    for day in range(int(numOfDayes)):
            day_key = "Day " + str(day+1)
            if day_key not in LLMPlan:
                LLMPlan[day_key] = []
    

    
    for (category, selections), (_, numEachDayByCategory) in zip(tags.items(), num_each_day_counts.items()):
        # print(f"{category}: {num}")
        totalNumberPerCategory = numEachDayByCategory * int(numOfDayes)
            
        query_result = query_weaviate(category, selections, totalNumberPerCategory, city)
        
        day_key= 1
        query_key = 0
        for i in range(int(numOfDayes)):
            for j in range(numEachDayByCategory):
                # print(category, query_key)
                if query_result["data"]["Get"][city] is None:
                    break
                if query_key >= len(query_result["data"]["Get"][city]):
                    break
                result = query_result["data"]["Get"][city][query_key]
                item = {
                    "name": result["name"],
                    "description": result["description"],
                    "bannerImage": result["bannerImage"],
                    "slugCategoryPOI": result["slugCategoryPOI"],
                    "slugCity": result["slugCity"],
                    "location": result["location"]
                }
                LLMItem = {
                    "name": result["name"],
                    "description": result["description"][0:150],
                    "slugCity": result["slugCity"],
                }
                random_index = random.randint(0, len(TripPlan["Day " + str(day_key)]))
                TripPlan["Day " + str(day_key)].insert(random_index, item)
                LLMPlan["Day " + str(day_key)].insert(random_index, LLMItem)
                # TripPlan["Day " + str(day_key)].append(item)
                # LLMPlan["Day " + str(day_key)].append(LLMItem)
                query_key += 1
            # if TripPlan["Day " + str(day_key)] == []:
            #     del TripPlan["Day " + str(day_key)]
            #     del LLMPlan["Day " + str(day_key)]
            #     continue
            day_key += 1

    if TripPlan["Day " + "1"] == []:
        emptyCase = {
            "Attractions": ["Family & Kids", "Amusement Parks", "Cultural & Historical", "Iconic Landmarks", "Parks", "Nature & wildlife"],
            "Restaurants": [],
            "Shopping": []
        }
        return itinerary(city, numOfDayes, emptyCase)
    
    for day in range(int(numOfDayes)):
            day_key = "Day " + str(day+1)
            if TripPlan[day_key] == []:
                del TripPlan[day_key]
                del LLMPlan[day_key]


    start_time = time.time()
    # LLMDes = LLMDescription(LLMPlan)
    LLMDes = LLMCompletions(LLMPlan)
    end_time = time.time()
    print("Time taken to generate LLM Description: ", end_time - start_time)


    return TripPlan, LLMDes

    

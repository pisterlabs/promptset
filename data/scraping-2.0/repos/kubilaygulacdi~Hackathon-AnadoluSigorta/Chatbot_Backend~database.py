import pymongo
import requests
import time
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

myclient = pymongo.MongoClient("mongodb+srv://mkgulacdi:KDumYPV2ueOXINCn@ascluster.6rdtmya.mongodb.net/?retryWrites=true&w=majority")

db = myclient.get_database("complaint_db")
collection = db["customers"]

api_url = "http://127.0.0.1:5000/complaint/user"


last_unique_no = set()

while True:
    response = requests.get(api_url)

    if response.status_code == 200:
        json_data = response.json()
        current_unique_no = json_data.get("user").get("no")

        if current_unique_no not in last_unique_no:

            model = OpenAI(model="text-davinci-003",
                           openai_api_key="API-KEY")

            prompt = """
                Aşağıda kullanıcı tarafında yapılan şikayet yorumunun konusunu 3 kelime ile sınıflandır.
                İki çizgi arasındaki metni incele.
                ---
                {complaint}
                ---
                """

            prompt_template = PromptTemplate(input_variables=["complaint"], template=prompt)
            result = model(prompt_template.format(complaint=json_data["user"]["complaint"]))

            json_data["user"]["topic"] = result.strip()
            collection.insert_one(json_data)
            print("Başarıyla eklendi...")
            last_unique_no.add(current_unique_no)
        else:
            print("Alınan veri db içerisinde var...")
    else:
        print(f"Hata: {response.status_code}")


    time.sleep(15)


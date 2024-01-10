import weaviate
import json
from api_keys import OPENAI_API_KEY

auth_config = weaviate.AuthApiKey(api_key="readonly-demo")

client = weaviate.Client(
        url="https://edu-demo.weaviate.network",
        auth_client_secret= auth_config,
        additional_headers= {
            "X-OpenAI-Api-Key" : OPENAI_API_KEY
            }
        )

res = client.query.get("WikiCity", ["city_name", "wiki_summary"]).with_near_text(
        {
            "concepts" : ["Popular Southeast Asian Tourist Destination"]
            }).with_limit(3).with_generate(
                    single_prompt = "Write a tweet with a potentially surprising fact from {wiki_summary}"
                    ).do()

with open("tourism.txt", "w") as file:
    for city_result in res["data"]["Get"]["WikiCity"]:
        file.write(json.dumps(city_result["_additional"], indent=2))

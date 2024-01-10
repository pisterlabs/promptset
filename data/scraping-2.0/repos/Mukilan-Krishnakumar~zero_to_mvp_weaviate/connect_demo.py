import weaviate
import json
from api_keys import OPENAI_API_KEY

auth_config = weaviate.AuthApiKey(api_key="readonly-demo")

client = weaviate.Client(
        url= "https://edu-demo.weaviate.network",
        auth_client_secret= auth_config,
        additional_headers={
            "X-OpenAI-Api-Key": OPENAI_API_KEY
            }
        )


res = client.query.get(
        "WikiCity", ["city_name", "country", "lng", "lat"]
        ).with_near_text({
            "concepts" : ["Major Indian city"]
            }).with_limit(5).do()

print(json.dumps(res, indent=2))

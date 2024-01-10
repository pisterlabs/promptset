import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Moderation.create(
            input="How to become famous youtuber"
            )
print(response)

# {
#   "id": "modr-6mGfNL7Is8g9DRqYyofkmUVPzHvjt",
#   "model": "text-moderation-004",
#   "results": [
#     {
#       "categories": {
#         "hate": false,
#         "hate/threatening": false,
#         "self-harm": false,
#         "sexual": false,
#         "sexual/minors": false,
#         "violence": false,
#         "violence/graphic": false
#       },
#       "category_scores": {
#         "hate": 1.1264393151577679e-07,
#         "hate/threatening": 4.289473776175612e-10,
#         "self-harm": 1.5447776391397383e-10,
#         "sexual": 3.004195718858682e-07,
#         "sexual/minors": 4.454510360574204e-08,
#         "violence": 1.5531125541201618e-07,
#         "violence/graphic": 4.524354668689057e-09
#       },
#       "flagged": false
#     }
#   ]
# }